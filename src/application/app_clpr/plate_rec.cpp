#include "plate_rec.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/monopoly_allocator.hpp>
#include <common/preprocess_kernel.cuh>

namespace Clpr {
struct AffineMatrixRec {
    float i2d[6];
    float d2i[6];
    // Mat m2x3_d2i;
    // Mat m2x3_i2d;
    void compute(const float *landmarks, const cv::Size &net_size) {
        int h = net_size.height;
        int w = net_size.width;
        cv::Point2f pt1, pt2, pt3, pt4;
        pt1.x = landmarks[0];
        pt1.y = landmarks[1];
        pt2.x = landmarks[2];
        pt2.y = landmarks[3];
        pt3.x = landmarks[4];
        pt3.y = landmarks[5];
        pt4.x = landmarks[6];
        pt4.y = landmarks[7];

        cv::Point2f srcPts[4];
        srcPts[0] = pt1;
        srcPts[1] = pt2;
        srcPts[2] = pt3;
        srcPts[3] = pt4;
        cv::Point2f dstPts[4];
        dstPts[0] = cv::Point2f(0, 0);
        dstPts[1] = cv::Point2f(w, 0);
        dstPts[2] = cv::Point2f(w, h);
        dstPts[3] = cv::Point2f(0, h);
        // cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
        cv::Mat m2x3_i2d = cv::getAffineTransform(srcPts, dstPts);
        //  32F是必须的，
        m2x3_i2d.convertTo(m2x3_i2d, CV_32F);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    }

    cv::Mat i2d_mat() {
        return cv::Mat(2, 3, CV_32F, i2d);
    }
};
using RecControllerImpl = InferController<RecInput,            // input
                                          plateResult,         // output
                                          tuple<string, int>,  // start param
                                          AffineMatrixRec      // additional
                                          >;

class RecInferImpl : public RecInfer, public RecControllerImpl {
public:
    virtual ~RecInferImpl() {
        stop();
    }

    bool startup(const string &file, int gpuid) {
        return RecControllerImpl::startup(make_tuple(file, gpuid));
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
        int max_batch_size = engine->get_max_batch_size();
        auto input         = engine->tensor("images");
        auto output_rec    = engine->tensor("output_1");  // n x 21 x 78
        auto output_color  = engine->tensor("output_2");  // n x 5
        input_width_       = input->size(3);
        input_height_      = input->size(2);
        gpu_               = gpuid;
        tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        stream_            = engine->get_stream();
        result.set_value(true);
        input->resize_single_dim(0, max_batch_size);

        // int n = 0;
        vector<Job> fetch_jobs;
        while (get_jobs_and_wait(fetch_jobs, max_batch_size)) {
            int infer_batch_size = fetch_jobs.size();
            input->resize_single_dim(0, infer_batch_size);

            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto &job = fetch_jobs[ibatch];
                input->copy_from_gpu(input->offset(ibatch), job.mono_tensor->data()->gpu(), input->count(1));
                job.mono_tensor->release();
            }

            engine->forward(false);
            for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                auto &job         = fetch_jobs[ibatch];
                auto plate_result = job.output;

                // plate number decode
                float *image_based_output_rec = output_rec->cpu<float>(ibatch);
                auto s1                       = output_rec->size(1);  // 21
                auto s2                       = output_rec->size(2);  // 78

                string pre_str{};
                float number_socre{0.};
                for (int x = 0; x < s1; x++) {
                    auto max_value =
                        *max_element(image_based_output_rec + x * s2, image_based_output_rec + (x + 1) * s2);
                    auto index = max_element(image_based_output_rec + x * s2, image_based_output_rec + (x + 1) * s2) -
                                 (image_based_output_rec + x * s2);
                    // INFO("curren index is %d, value is %f", index, max_value);
                    if (CHARS[index] != "#" && CHARS[index] != pre_str) {
                        plate_result.number += CHARS[index];
                        number_socre += max_value;
                    }
                    pre_str = CHARS[index];
                }
                plate_result.number_confidence = number_socre / plate_result.number.size();
                // plate color

                float *image_based_output_color = output_color->cpu<float>(ibatch);
                // for (int z = 0; z < 5; z++) {
                //     INFO("color score is %f", image_based_output_color[z]);
                // }
                auto color_socre = *max_element(image_based_output_color, image_based_output_color + 5);
                int color_index =
                    max_element(image_based_output_color, image_based_output_color + 5) - image_based_output_color;
                plate_result.color            = static_cast<PlateColor>(color_index);
                plate_result.color_confidence = color_socre;
                //
                job.pro->set_value(plate_result);
            }
            fetch_jobs.clear();
        };
        stream_ = nullptr;
        tensor_allocator_.reset();
        INFO("Engine destroy.");
    }
    virtual shared_future<plateResult> commit(const RecInput &input) override {
        return RecControllerImpl::commit(input);
    }

    virtual vector<shared_future<plateResult>> commits(const vector<RecInput> &inputs) override {
        return RecControllerImpl::commits(inputs);
    }

    virtual bool preprocess(Job &job, const RecInput &input) override {
        if (tensor_allocator_ == nullptr) {
            INFOE("tensor_allocator_ is nullptr");
            return false;
        }
        auto &image = get<0>(input);
        auto box    = get<1>(input);
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
        auto &tensor = job.mono_tensor->data();
        if (tensor == nullptr) {
            // not init
            tensor = make_shared<TRT::Tensor>();
            tensor->set_workspace(make_shared<TRT::MixMemory>());
        }
        Size input_size(input_width_, input_height_);
        job.additional.compute(box, input_size);

        tensor->set_stream(stream_);
        tensor->resize(1, 3, input_height_, input_width_);

        size_t size_image           = image.cols * image.rows * 3;
        size_t size_matrix          = iLogger::upbound(sizeof(job.additional.d2i), 32);
        auto workspace              = tensor->get_workspace();
        uint8_t *gpu_workspace      = (uint8_t *)workspace->gpu(size_image + size_matrix);
        float *affine_matrix_device = (float *)gpu_workspace;
        uint8_t *image_device       = gpu_workspace + size_matrix;
        checkCudaRuntime(cudaMemcpyAsync(image_device, image.data, size_image, cudaMemcpyHostToDevice, stream_));
        checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, job.additional.d2i, sizeof(job.additional.d2i),
                                         cudaMemcpyHostToDevice, stream_));

        // auto normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0, 0, CUDAKernel::ChannelType::Invert);
        float mean[3]  = {0.588, 0.588, 0.588};
        float std[3]   = {0.193, 0.193, 0.193};
        auto normalize = CUDAKernel::Norm::mean_std(mean, std, 1 / 255.0, CUDAKernel::ChannelType::None);
        CUDAKernel::warp_affine_bilinear_and_normalize_plane(image_device, image.cols * 3, image.cols, image.rows,
                                                             tensor->gpu<float>(), input_width_, input_height_,
                                                             affine_matrix_device, 127, normalize, stream_);
        return true;
    }

private:
    int input_width_      = 0;
    int input_height_     = 0;
    int gpu_              = 0;
    TRT::CUStream stream_ = nullptr;
};

shared_ptr<RecInfer> create_rec(const string &engine_file, int gpuid) {
    shared_ptr<RecInferImpl> instance(new RecInferImpl());
    if (!instance->startup(engine_file, gpuid)) {
        instance.reset();
    }
    return instance;
}

}  // namespace Clpr