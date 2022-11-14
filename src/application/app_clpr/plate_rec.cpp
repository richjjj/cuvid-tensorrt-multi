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
using RecControllerImpl = InferController<Input,               // input
                                          plateNO,             // output
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
        auto output        = engine->tensor("output");  // b x 21
        auto output_size   = output->size(1);
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
                auto &job                 = fetch_jobs[ibatch];
                float *image_based_output = output->cpu<float>(ibatch);
                auto plate_str            = job.output;
                // decode
                string pre_str = "";
                for (int x = 0; x < output_size; x++) {
                    int index = image_based_output[x];
                    if (CHARS[index] != "#" && CHARS[index] != pre_str) {
                        plate_str += CHARS[index];
                    }
                    pre_str = CHARS[index];
                }
                job.pro->set_value(plate_str);
            }
            fetch_jobs.clear();
        };
        stream_ = nullptr;
        tensor_allocator_.reset();
        INFO("Engine destroy.");
    }
    virtual shared_future<plateNO> commit(const Input &input) override {
        return RecControllerImpl::commit(input);
    }

    virtual vector<shared_future<plateNO>> commits(const vector<Input> &inputs) override {
        return RecControllerImpl::commits(inputs);
    }

    virtual bool preprocess(Job &job, const Input &input) override {
        if (tensor_allocator_ == nullptr) {
            INFOE("tensor_allocator_ is nullptr");
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

        auto &image = get<0>(input);
        auto box    = get<1>(input);
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