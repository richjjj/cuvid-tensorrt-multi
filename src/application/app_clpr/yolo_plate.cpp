#include "yolo_plate.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>

namespace Clpr {

void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
                           float *invert_affine_matrix, float *parray, int max_objects, cudaStream_t stream);
void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, cudaStream_t stream);
struct AffineMatrix {
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const cv::Size &image_size, const cv::Rect &box, const cv::Size &net_size) {
        Rect box_ = box;
        if (box_.width == 0 || box_.height == 0) {
            box_.width  = image_size.width;
            box_.height = image_size.height;
            box_.x      = 0;
            box_.y      = 0;
        }

        float rate       = box_.width > 100 ? 0.1f : 0.15f;
        float pad_width  = box_.width * (1 + 2 * rate);
        float pad_height = box_.height * (1 + 1 * rate);
        float scale      = min(net_size.width / pad_width, net_size.height / pad_height);
        i2d[0]           = scale;
        i2d[1]           = 0;
        i2d[2] = -(box_.x - box_.width * 1 * rate + pad_width * 0.5) * scale + net_size.width * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] =
            -(box_.y - box_.height * 1 * rate + pad_height * 0.5) * scale + net_size.height * 0.5 + scale * 0.5 - 0.5;

        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    }

    cv::Mat i2d_mat() {
        return cv::Mat(2, 3, CV_32F, i2d);
    }
};
using DetControllerImpl = InferController<DetInput,            // input
                                          PlateRegionArray,    // output
                                          tuple<string, int>,  // start param
                                          AffineMatrix         // additional
                                          >;
class DetInferImpl : public DetInfer, public DetControllerImpl {
public:
    /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
    virtual ~DetInferImpl() {
        stop();
    }

    virtual bool startup(const string &file, int gpuid, float confidence_threshold, float nms_threshold,
                         int max_objects) {
        normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);

        confidence_threshold_ = confidence_threshold;
        nms_threshold_        = nms_threshold;
        max_objects_          = max_objects;
        return DetControllerImpl::startup(make_tuple(file, gpuid));
    }

    virtual void worker(promise<bool> &result) override {
        string file = get<0>(start_param_);
        int gpuid   = get<1>(start_param_);

        TRT::set_device(gpuid);
        auto engine = TRT::load_infer(file);
        if (engine == nullptr) {
            INFOE("Engine %s load failed", file.c_str());
            result.set_value(false);
            return;
        }

        engine->print();

        const int MAX_IMAGE_BBOX  = max_objects_;
        const int NUM_BOX_ELEMENT = 7 + 8;  // left, top, right, bottom, confidence, class, keepflag + 4*2
        TRT::Tensor affin_matrix_device(TRT::DataType::Float);
        TRT::Tensor output_array_device(TRT::DataType::Float);
        int max_batch_size = engine->get_max_batch_size();
        auto input         = engine->tensor("images");
        auto output        = engine->tensor("output");
        int num_classes    = output->size(2) - 5 - 8;  // num_classes = 2

        input_width_      = input->size(3);
        input_height_     = input->size(2);
        tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        stream_           = engine->get_stream();
        gpu_              = gpuid;
        result.set_value(true);

        input->resize_single_dim(0, max_batch_size).to_gpu();
        affin_matrix_device.set_stream(stream_);

        // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
        affin_matrix_device.resize(max_batch_size, 8).to_gpu();

        // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
        output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();

        vector<Job> fetch_jobs;
        while (get_jobs_and_wait(fetch_jobs, max_batch_size)) {
            int infer_batch_size = fetch_jobs.size();
            input->resize_single_dim(0, infer_batch_size);

            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto &job  = fetch_jobs[ibatch];
                auto &mono = job.mono_tensor->data();
                CUDATools::AutoDevice auto_device_exchange(mono->device());

                if (mono->get_stream() != stream_) {
                    // synchronize preprocess stream finish
                    checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                }

                affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                job.mono_tensor->release();
            }

            engine->forward(false);
            output_array_device.to_gpu(false);
            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto &job                 = fetch_jobs[ibatch];
                float *image_based_output = output->gpu<float>(ibatch);
                float *output_array_ptr   = output_array_device.gpu<float>(ibatch);
                auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);
                checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                decode_kernel_invoker(image_based_output, output->size(1), num_classes, confidence_threshold_,
                                      affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);
                nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
            }

            output_array_device.to_cpu();
            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                float *parray           = output_array_device.cpu<float>(ibatch);
                int count               = min(MAX_IMAGE_BBOX, (int)*parray);
                auto &job               = fetch_jobs[ibatch];
                auto &image_based_boxes = job.output;
                for (int i = 0; i < count; ++i) {
                    float *pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                    int label    = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1) {
                        image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label, pbox + 7);
                    }
                }
                job.pro->set_value(image_based_boxes);
            }
            fetch_jobs.clear();
        }
        stream_ = nullptr;
        tensor_allocator_.reset();
        INFO("Engine destroy.");
    }

    virtual bool preprocess(Job &job, const DetInput &detinput) override {
        if (tensor_allocator_ == nullptr) {
            INFOE("tensor_allocator_ is nullptr");
            return false;
        }
        auto &image = detinput.image;
        if (image.empty()) {
            INFOE("Image is empty");
            return false;
        }

        job.mono_tensor = tensor_allocator_->query();
        if (job.mono_tensor == nullptr) {
            INFOE("Tensor allocator query failed.");
            return false;
        }

        CUDATools::AutoDevice auto_device(gpu_);
        auto &tensor                    = job.mono_tensor->data();
        TRT::CUStream preprocess_stream = nullptr;

        if (tensor == nullptr) {
            // not init
            tensor = make_shared<TRT::Tensor>();
            tensor->set_workspace(make_shared<TRT::MixMemory>());

            // owner = false, tensor ignored the stream
            tensor->set_stream(stream_, false);
        }

        preprocess_stream = tensor->get_stream();
        Size input_size(input_width_, input_height_);
        job.additional.compute(image.size(), detinput.roiRect, input_size);
        tensor->resize(1, 3, input_height_, input_width_);

        size_t size_image           = image.cols * image.rows * 3;
        size_t size_matrix          = iLogger::upbound(sizeof(job.additional.d2i), 32);
        auto workspace              = tensor->get_workspace();
        uint8_t *gpu_workspace      = (uint8_t *)workspace->gpu(size_matrix + size_image);
        float *affine_matrix_device = (float *)gpu_workspace;
        uint8_t *image_device       = size_matrix + gpu_workspace;

        checkCudaRuntime(
            cudaMemcpyAsync(image_device, image.data, size_image, cudaMemcpyHostToDevice, preprocess_stream));

        checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, job.additional.d2i, sizeof(job.additional.d2i),
                                         cudaMemcpyHostToDevice, preprocess_stream));

        CUDAKernel::warp_affine_bilinear_and_normalize_plane(image_device, image.cols * 3, image.cols, image.rows,
                                                             tensor->gpu<float>(), input_width_, input_height_,
                                                             affine_matrix_device, 114, normalize_, preprocess_stream);

        return true;
    }

    virtual vector<shared_future<PlateRegionArray>> commits(const vector<DetInput> &images) override {
        return DetControllerImpl::commits(images);
    }

    virtual shared_future<PlateRegionArray> commit(const DetInput &image) override {
        return DetControllerImpl::commit(image);
    }
    virtual vector<shared_future<PlateRegionArray>> commits(const vector<Mat> &images) override {
        vector<DetInput> tmp;
        tmp.reserve(images.size());
        for (auto &im : images) {
            tmp.emplace_back(im);
        }
        //         std::vector<...> attributes;
        // attributes.reserve(instances.size());
        // std::transform(instances.begin(), instances.end(), std::back_inserter(attributes),
        //                [](auto&& obj) { return obj.a; });
        return DetControllerImpl::commits(tmp);
    }

private:
    int input_width_            = 0;
    int input_height_           = 0;
    int gpu_                    = 0;
    float confidence_threshold_ = 0;
    float nms_threshold_        = 0;
    int max_objects_            = 1024;
    TRT::CUStream stream_       = nullptr;
    CUDAKernel::Norm normalize_;
};

shared_ptr<DetInfer> create_det(const string &engine_file, int gpuid, float confidence_threshold, float nms_threshold,
                                int max_objects) {
    shared_ptr<DetInferImpl> instance(new DetInferImpl());
    if (!instance->startup(engine_file, gpuid, confidence_threshold, nms_threshold, max_objects)) {
        instance.reset();
    }
    return instance;
}
void image_to_tensor(const cv::Mat &image, shared_ptr<TRT::Tensor> &tensor, int ibatch) {
    CUDAKernel::Norm normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);

    Size input_size(tensor->size(3), tensor->size(2));
    AffineMatrix affine;
    affine.compute(image.size(), Rect(0, 0, image.cols, image.rows), input_size);

    size_t size_image           = image.cols * image.rows * 3;
    size_t size_matrix          = iLogger::upbound(sizeof(affine.d2i), 32);
    auto workspace              = tensor->get_workspace();
    uint8_t *gpu_workspace      = (uint8_t *)workspace->gpu(size_matrix + size_image);
    float *affine_matrix_device = (float *)gpu_workspace;
    uint8_t *image_device       = size_matrix + gpu_workspace;

    uint8_t *cpu_workspace    = (uint8_t *)workspace->cpu(size_matrix + size_image);
    float *affine_matrix_host = (float *)cpu_workspace;
    uint8_t *image_host       = size_matrix + cpu_workspace;
    auto stream               = tensor->get_stream();

    memcpy(image_host, image.data, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
    checkCudaRuntime(
        cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

    CUDAKernel::warp_affine_bilinear_and_normalize_plane(
        image_device, image.cols * 3, image.cols, image.rows, tensor->gpu<float>(ibatch), input_size.width,
        input_size.height, affine_matrix_device, 114, normalize, stream);
    tensor->synchronize();
}
}  // namespace Clpr