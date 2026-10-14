#include "mnn_wrapper.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Executor.hpp>

#include <cstring>
#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <memory>

namespace MNN
{
    class RuntimeCreator;
    MNN_PUBLIC const RuntimeCreator *MNNGetExtraRuntimeCreator(MNNForwardType type);
}

// C++11 compatible make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique_ptr(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Global mutex to serialize MNN inference calls
// MNN's internal thread pool has a limit of MNN_THREAD_POOL_MAX_TASKS (default=2)
// This mutex ensures only one inference runs at a time, avoiding thread pool exhaustion
static std::mutex g_mnn_inference_mutex;

// ============== Internal Structures ==============

struct MNN_SharedRuntime
{
    MNN::BackendConfig backend_config;
    MNN::ScheduleConfig schedule_config;
    int thread_count;
    int precision_mode;
};

struct MNN_InferenceEngine
{
    std::unique_ptr<MNN::Interpreter> interpreter;
    MNN::Session *default_session;
    std::mutex mutex;
    std::string last_error;

    std::vector<int> input_shape;
    std::vector<int> output_shape;
    MNN::Tensor *input_tensor;
    MNN::Tensor *output_tensor;

    MNN_SharedRuntime *runtime; // Optional shared runtime
    bool owns_runtime;

    MNN_InferenceEngine() : default_session(nullptr), input_tensor(nullptr),
                            output_tensor(nullptr), runtime(nullptr), owns_runtime(false) {}
};

struct MNN_SingleSession
{
    MNN::Session *session;
    MNN_InferenceEngine *engine;
    std::string last_error;
    MNN::Tensor *input_tensor;
    MNN::Tensor *output_tensor;

    MNN_SingleSession() : session(nullptr), engine(nullptr),
                          input_tensor(nullptr), output_tensor(nullptr) {}
};

struct MNN_SessionPool
{
    MNN_InferenceEngine *engine;
    std::vector<MNN::Session *> sessions;
    std::vector<MNN::Tensor *> input_tensors;
    std::vector<MNN::Tensor *> output_tensors;

    std::mutex mutex;
    std::condition_variable cv;
    std::queue<size_t> available_sessions;
    std::string last_error;
};

// ============== Helper Functions ==============

// Initialize schedule and backend configs from MNNR_Config.
// Caller must ensure `schedule` and `backend` outlive any use of schedule.backendConfig.
static void init_schedule_config(MNN::ScheduleConfig &schedule, MNN::BackendConfig &backend, const MNNR_Config *config)
{
    schedule.type = (config) ? static_cast<MNNForwardType>(config->forward_type) : MNN_FORWARD_CPU;
    if (config && config->gpu_mode != 0 &&
        (schedule.type == MNN_FORWARD_OPENCL || schedule.type == MNN_FORWARD_VULKAN))
    {
        schedule.mode = config->gpu_mode;
    }
    else
    {
        schedule.numThread = config ? config->thread_count : 4;
        if (schedule.numThread <= 0)
        {
            schedule.numThread = 4;
        }
    }

    if (config)
    {
        switch (config->precision_mode)
        {
        case 1:
            backend.precision = MNN::BackendConfig::Precision_Low;
            break;
        case 2:
            backend.precision = MNN::BackendConfig::Precision_High;
            break;
        default:
            backend.precision = MNN::BackendConfig::Precision_Normal;
            break;
        }
    }
    schedule.backendConfig = &backend;
}

static bool init_engine_tensors(MNN_InferenceEngine *engine)
{
    if (!engine->interpreter || !engine->default_session)
    {
        return false;
    }

    // Get input tensor
    auto input_map = engine->interpreter->getSessionInputAll(engine->default_session);
    if (input_map.empty())
    {
        engine->last_error = "No input tensors found";
        return false;
    }

    engine->input_tensor = input_map.begin()->second;
    auto input_shape_vec = engine->input_tensor->shape();
    engine->input_shape.assign(input_shape_vec.begin(), input_shape_vec.end());

    // Get output tensor
    auto output_map = engine->interpreter->getSessionOutputAll(engine->default_session);
    if (output_map.empty())
    {
        engine->last_error = "No output tensors found";
        return false;
    }

    engine->output_tensor = output_map.begin()->second;
    auto output_shape_vec = engine->output_tensor->shape();
    engine->output_shape.assign(output_shape_vec.begin(), output_shape_vec.end());

    return true;
}

// ============== Version & Info ==============

const char *mnnr_get_version(void)
{
    return MNN_VERSION;
}

bool mnnr_is_backend_available(int32_t forward_type)
{
    if (forward_type < MNN_FORWARD_CPU || forward_type >= MNN_FORWARD_ALL)
    {
        return false;
    }
    return MNN::MNNGetExtraRuntimeCreator(static_cast<MNNForwardType>(forward_type)) != nullptr;
}

// ============== Shared Runtime API ==============

MNN_SharedRuntime *mnnr_create_runtime(const MNNR_Config *config)
{
    auto runtime = new MNN_SharedRuntime();

    runtime->thread_count = config ? config->thread_count : 4;
    if (runtime->thread_count <= 0)
    {
        runtime->thread_count = 4;
    }

    runtime->precision_mode = config ? config->precision_mode : 0;

    init_schedule_config(runtime->schedule_config, runtime->backend_config, config);

    return runtime;
}

void mnnr_destroy_runtime(MNN_SharedRuntime *runtime)
{
    delete runtime;
}

// ============== Inference Engine API ==============

MNN_InferenceEngine *mnnr_create_engine(
    const void *buffer,
    size_t size,
    const MNNR_Config *config)
{
    if (!buffer || size == 0)
    {
        return nullptr;
    }

    auto engine = new MNN_InferenceEngine();

    // Create interpreter from buffer
    engine->interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    if (!engine->interpreter)
    {
        engine->last_error = "Failed to create interpreter from buffer";
        delete engine;
        return nullptr;
    }

    // Create default session
    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    engine->default_session = engine->interpreter->createSession(schedule);
    if (!engine->default_session)
    {
        engine->last_error = "Failed to create default session";
        delete engine;
        return nullptr;
    }

    // Initialize tensors
    if (!init_engine_tensors(engine))
    {
        delete engine;
        return nullptr;
    }

    return engine;
}

MNN_InferenceEngine *mnnr_create_engine_with_runtime(
    const void *buffer,
    size_t size,
    MNN_SharedRuntime *runtime)
{
    if (!buffer || size == 0 || !runtime)
    {
        return nullptr;
    }

    auto engine = new MNN_InferenceEngine();
    engine->runtime = runtime;
    engine->owns_runtime = false;

    // Create interpreter from buffer
    engine->interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    if (!engine->interpreter)
    {
        engine->last_error = "Failed to create interpreter from buffer";
        delete engine;
        return nullptr;
    }

    // Create session using shared runtime config
    engine->default_session = engine->interpreter->createSession(runtime->schedule_config);
    if (!engine->default_session)
    {
        engine->last_error = "Failed to create session with shared runtime";
        delete engine;
        return nullptr;
    }

    // Initialize tensors
    if (!init_engine_tensors(engine))
    {
        delete engine;
        return nullptr;
    }

    return engine;
}

void mnnr_destroy_engine(MNN_InferenceEngine *engine)
{
    if (engine)
    {
        if (engine->default_session && engine->interpreter)
        {
            engine->interpreter->releaseSession(engine->default_session);
        }
        delete engine;
    }
}

MNNR_ErrorCode mnnr_get_input_shape(
    const MNN_InferenceEngine *engine,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    *out_ndims = engine->input_shape.size();
    for (size_t i = 0; i < engine->input_shape.size() && i < 8; i++)
    {
        dims[i] = static_cast<size_t>(engine->input_shape[i]);
    }

    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_get_output_shape(
    const MNN_InferenceEngine *engine,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    *out_ndims = engine->output_shape.size();
    for (size_t i = 0; i < engine->output_shape.size() && i < 8; i++)
    {
        dims[i] = static_cast<size_t>(engine->output_shape[i]);
    }

    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_run_inference(
    MNN_InferenceEngine *engine,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!engine || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Use global lock to serialize MNN inference (thread pool limit)
    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);

    // Calculate expected sizes
    size_t expected_input = 1;
    for (int dim : engine->input_shape)
    {
        expected_input *= dim;
    }

    size_t expected_output = 1;
    for (int dim : engine->output_shape)
    {
        expected_output *= dim;
    }

    if (input_size != expected_input || output_size != expected_output)
    {
        engine->last_error = "Input/output size mismatch";
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Create host tensor and copy input data
    auto input_host = make_unique_ptr<MNN::Tensor>(engine->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));
    engine->input_tensor->copyFromHostTensor(input_host.get());

    // Run inference
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    // Copy output data
    auto output_host = make_unique_ptr<MNN::Tensor>(engine->output_tensor, MNN::Tensor::CAFFE);
    engine->output_tensor->copyToHostTensor(output_host.get());
    std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));

    return MNNR_SUCCESS;
}

const char *mnnr_get_last_error(const MNN_InferenceEngine *engine)
{
    if (!engine)
    {
        return "Engine is null";
    }
    return engine->last_error.c_str();
}

// ============== Session Pool API ==============

MNN_SessionPool *mnnr_create_session_pool(
    MNN_InferenceEngine *engine,
    size_t pool_size,
    const MNNR_Config *config)
{
    if (!engine || pool_size == 0)
    {
        return nullptr;
    }

    auto pool = new MNN_SessionPool();
    pool->engine = engine;

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);

    // Create sessions
    for (size_t i = 0; i < pool_size; i++)
    {
        MNN::Session *session = engine->interpreter->createSession(schedule);
        if (!session)
        {
            // Cleanup on failure
            for (auto s : pool->sessions)
            {
                engine->interpreter->releaseSession(s);
            }
            // Note: input/output tensors are owned by MNN sessions, not by us.
            // They will be freed when sessions are released above.
            delete pool;
            return nullptr;
        }

        pool->sessions.push_back(session);
        pool->available_sessions.push(i);

        // Get input/output tensors for this session
        auto input_map = engine->interpreter->getSessionInputAll(session);
        auto output_map = engine->interpreter->getSessionOutputAll(session);

        pool->input_tensors.push_back(input_map.begin()->second);
        pool->output_tensors.push_back(output_map.begin()->second);
    }

    return pool;
}

void mnnr_destroy_session_pool(MNN_SessionPool *pool)
{
    if (pool)
    {
        for (auto session : pool->sessions)
        {
            if (pool->engine && pool->engine->interpreter)
            {
                pool->engine->interpreter->releaseSession(session);
            }
        }
        delete pool;
    }
}

MNNR_ErrorCode mnnr_session_pool_run(
    MNN_SessionPool *pool,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!pool || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Acquire a session (this will block if all sessions are busy)
    size_t session_idx;
    {
        std::unique_lock<std::mutex> lock(pool->mutex);
        pool->cv.wait(lock, [pool]
                      { return !pool->available_sessions.empty(); });
        session_idx = pool->available_sessions.front();
        pool->available_sessions.pop();
    }

    // Run inference with global lock to serialize MNN thread pool access
    MNNR_ErrorCode result = MNNR_SUCCESS;

    auto *session = pool->sessions[session_idx];
    auto *input_tensor = pool->input_tensors[session_idx];
    auto *output_tensor = pool->output_tensors[session_idx];

    // Create host tensor and copy input (can be done outside the global lock)
    auto input_host = make_unique_ptr<MNN::Tensor>(input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));

    {
        // Global lock for MNN inference to avoid thread pool exhaustion
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

        input_tensor->copyFromHostTensor(input_host.get());

        // Run inference
        MNN::ErrorCode code = pool->engine->interpreter->runSession(session);
        if (code != MNN::NO_ERROR)
        {
            pool->last_error = "Session pool inference failed";
            result = MNNR_ERROR_RUNTIME_ERROR;
        }
        else
        {
            // Copy output
            auto output_host = make_unique_ptr<MNN::Tensor>(output_tensor, MNN::Tensor::CAFFE);
            output_tensor->copyToHostTensor(output_host.get());
            std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));
        }
    }

    // Release session
    {
        std::lock_guard<std::mutex> lock(pool->mutex);
        pool->available_sessions.push(session_idx);
    }
    pool->cv.notify_one();

    return result;
}

size_t mnnr_session_pool_available(const MNN_SessionPool *pool)
{
    if (!pool)
    {
        return 0;
    }
    std::lock_guard<std::mutex> lock(const_cast<MNN_SessionPool *>(pool)->mutex);
    return pool->available_sessions.size();
}

const char *mnnr_session_pool_get_last_error(const MNN_SessionPool *pool)
{
    if (!pool)
    {
        return "Pool is null";
    }
    return pool->last_error.c_str();
}

// ============== Single Session API ==============

MNN_SingleSession *mnnr_create_session(
    MNN_InferenceEngine *engine,
    const MNNR_Config *config)
{
    if (!engine)
    {
        return nullptr;
    }

    auto session = new MNN_SingleSession();
    session->engine = engine;

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    session->session = engine->interpreter->createSession(schedule);

    if (!session->session)
    {
        delete session;
        return nullptr;
    }

    // Get tensors
    auto input_map = engine->interpreter->getSessionInputAll(session->session);
    auto output_map = engine->interpreter->getSessionOutputAll(session->session);

    if (input_map.empty() || output_map.empty())
    {
        engine->interpreter->releaseSession(session->session);
        delete session;
        return nullptr;
    }

    session->input_tensor = input_map.begin()->second;
    session->output_tensor = output_map.begin()->second;

    return session;
}

void mnnr_destroy_session(MNN_SingleSession *session)
{
    if (session)
    {
        if (session->session && session->engine && session->engine->interpreter)
        {
            session->engine->interpreter->releaseSession(session->session);
        }
        delete session;
    }
}

MNNR_ErrorCode mnnr_run_inference_with_session(
    MNN_SingleSession *session,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!session || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Create host tensor and copy input (outside lock)
    auto input_host = make_unique_ptr<MNN::Tensor>(session->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));

    {
        // Global lock for MNN inference to avoid thread pool exhaustion
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

        session->input_tensor->copyFromHostTensor(input_host.get());

        // Run inference
        MNN::ErrorCode code = session->engine->interpreter->runSession(session->session);
        if (code != MNN::NO_ERROR)
        {
            session->last_error = "Session inference failed";
            return MNNR_ERROR_RUNTIME_ERROR;
        }

        // Copy output
        auto output_host = make_unique_ptr<MNN::Tensor>(session->output_tensor, MNN::Tensor::CAFFE);
        session->output_tensor->copyToHostTensor(output_host.get());
        std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));
    }

    return MNNR_SUCCESS;
}

const char *mnnr_session_get_last_error(const MNN_SingleSession *session)
{
    if (!session)
    {
        return "Session is null";
    }
    return session->last_error.c_str();
}

// ============== Dynamic Shape API ==============

MNNR_ErrorCode mnnr_run_inference_dynamic(
    MNN_InferenceEngine *engine,
    const float *input_data,
    const size_t *input_dims,
    size_t input_ndims,
    float **output_data,
    size_t *output_size,
    size_t *output_dims,
    size_t *output_ndims)
{
    if (!engine || !input_data || !input_dims || !output_data || !output_size || !output_dims || !output_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);

    // Build new input shape
    std::vector<int> new_shape(input_ndims);
    size_t total_input_size = 1;
    for (size_t i = 0; i < input_ndims; i++)
    {
        new_shape[i] = static_cast<int>(input_dims[i]);
        total_input_size *= input_dims[i];
    }

    // Resize input tensor
    engine->interpreter->resizeTensor(engine->input_tensor, new_shape);
    engine->interpreter->resizeSession(engine->default_session);

    // Get the updated input tensor after resize
    auto input_map = engine->interpreter->getSessionInputAll(engine->default_session);
    if (input_map.empty())
    {
        engine->last_error = "No input tensors found after resize";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    engine->input_tensor = input_map.begin()->second;

    // Create host tensor and copy input data
    auto input_host = make_unique_ptr<MNN::Tensor>(engine->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, total_input_size * sizeof(float));
    engine->input_tensor->copyFromHostTensor(input_host.get());

    // Run inference
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Dynamic inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    // Get output tensor after inference
    auto output_map = engine->interpreter->getSessionOutputAll(engine->default_session);
    if (output_map.empty())
    {
        engine->last_error = "No output tensors found";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    engine->output_tensor = output_map.begin()->second;

    // Get output shape
    auto output_shape = engine->output_tensor->shape();
    *output_ndims = output_shape.size();
    size_t total_output_size = 1;
    for (size_t i = 0; i < output_shape.size() && i < 8; i++)
    {
        output_dims[i] = static_cast<size_t>(output_shape[i]);
        total_output_size *= output_shape[i];
    }
    *output_size = total_output_size;

    // Allocate output buffer
    *output_data = new float[total_output_size];

    // Copy output data
    auto output_host = make_unique_ptr<MNN::Tensor>(engine->output_tensor, MNN::Tensor::CAFFE);
    engine->output_tensor->copyToHostTensor(output_host.get());
    std::memcpy(*output_data, output_host->host<float>(), total_output_size * sizeof(float));

    return MNNR_SUCCESS;
}

void mnnr_free_output(float *output_data)
{
    delete[] output_data;
}

// ============== Multi-Input/Output Session API ==============

// Implemented on the MNN Express Module API — the same execution path as
// MNN's Python `nn.load_module_from_file` (placeholder dtypes, dynamic
// shapes). Input/output names are discovered from the model file.
struct MNN_MultiSession
{
    std::mutex mutex;
    std::string last_error;

    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime;
    std::shared_ptr<MNN::Express::Module> module;

    // Discovered model I/O metadata (module input order = discovery order)
    std::vector<std::string> input_names;
    std::vector<halide_type_t> input_types;
    std::vector<std::string> output_names;

    // Staged inputs: raw bytes (both f32 and i32 elements are 4 bytes)
    std::map<std::string, std::vector<uint8_t>> staged_data;
    std::map<std::string, std::vector<int>> staged_dims;

    // Outputs of the last run, keyed by name
    std::map<std::string, MNN::Express::VARP> last_outputs;
};

static int64_t multi_prod(const std::vector<int> &dims)
{
    int64_t n = 1;
    for (int d : dims)
    {
        n *= d;
    }
    return n;
}

// Build the runtime manager + load the module for a session whose name lists
// and input dtypes are already filled in.
static MNNR_ErrorCode multi_finalize(MNN_MultiSession *s, const void *buffer, size_t size, const MNNR_Config *config)
{
    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    s->runtime.reset(MNN::Express::Executor::RuntimeManager::createRuntimeManager(schedule),
                     MNN::Express::Executor::RuntimeManager::destroy);
    if (!s->runtime)
    {
        s->last_error = "failed to create runtime manager";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    MNN::Express::Module::Config module_config;
    module_config.shapeMutable = true;
    s->module.reset(MNN::Express::Module::load(
        s->input_names, s->output_names, static_cast<const uint8_t *>(buffer), size,
        s->runtime, &module_config));
    if (!s->module)
    {
        s->last_error = "Module::load failed";
        return MNNR_ERROR_MODEL_LOAD_FAILED;
    }
    return MNNR_SUCCESS;
}

MNN_MultiSession *mnnr_multi_create(const void *buffer, size_t size, const MNNR_Config *config)
{
    if (!buffer || size == 0)
    {
        return nullptr;
    }

    auto s = new MNN_MultiSession();

    // Discover input/output names + declared input dtypes via a throwaway
    // Session parse. NOTE: the map loses the model graph's original input
    // order — prefer mnnr_multi_create_named when the order is known.
    {
        std::unique_ptr<MNN::Interpreter> interp(MNN::Interpreter::createFromBuffer(buffer, size));
        if (!interp)
        {
            delete s;
            return nullptr;
        }
        MNN::ScheduleConfig schedule;
        MNN::BackendConfig backend;
        init_schedule_config(schedule, backend, config);
        MNN::Session *probe = interp->createSession(schedule);
        if (!probe)
        {
            delete s;
            return nullptr;
        }
        for (const auto &kv : interp->getSessionInputAll(probe))
        {
            s->input_names.push_back(kv.first);
            s->input_types.push_back(kv.second->getType());
        }
        for (const auto &kv : interp->getSessionOutputAll(probe))
        {
            s->output_names.push_back(kv.first);
        }
    }
    if (s->input_names.empty() || s->output_names.empty())
    {
        delete s;
        return nullptr;
    }

    MNNR_ErrorCode code = multi_finalize(s, buffer, size, config);
    if (code != MNNR_SUCCESS)
    {
        delete s;
        return nullptr;
    }
    return s;
}

MNN_MultiSession *mnnr_multi_create_named(
    const void *buffer,
    size_t size,
    const MNNR_Config *config,
    const char *const *input_names,
    int32_t n_inputs,
    const char *const *output_names,
    int32_t n_outputs)
{
    if (!buffer || size == 0 || !input_names || !output_names || n_inputs <= 0 || n_outputs <= 0)
    {
        return nullptr;
    }

    auto s = new MNN_MultiSession();
    for (int32_t i = 0; i < n_inputs; i++)
    {
        s->input_names.push_back(input_names[i]);
    }
    for (int32_t i = 0; i < n_outputs; i++)
    {
        s->output_names.push_back(output_names[i]);
    }

    // Collect declared input dtypes (by name) via a throwaway Session parse
    {
        std::unique_ptr<MNN::Interpreter> interp(MNN::Interpreter::createFromBuffer(buffer, size));
        if (!interp)
        {
            delete s;
            return nullptr;
        }
        MNN::ScheduleConfig schedule;
        MNN::BackendConfig backend;
        init_schedule_config(schedule, backend, config);
        MNN::Session *probe = interp->createSession(schedule);
        if (!probe)
        {
            delete s;
            return nullptr;
        }
        auto all = interp->getSessionInputAll(probe);
        for (const auto &name : s->input_names)
        {
            auto it = all.find(name);
            if (it == all.end())
            {
                s->last_error = "input tensor not found: " + name;
                delete s;
                return nullptr;
            }
            s->input_types.push_back(it->second->getType());
        }
    }

    MNNR_ErrorCode code = multi_finalize(s, buffer, size, config);
    if (code != MNNR_SUCCESS)
    {
        delete s;
        return nullptr;
    }
    return s;
}

void mnnr_multi_destroy(MNN_MultiSession *session)
{
    delete session;
}

MNNR_ErrorCode mnnr_multi_set_input(
    MNN_MultiSession *session,
    const char *name,
    const int32_t *dims,
    int32_t ndims,
    const void *data,
    int32_t dtype)
{
    if (!session || !name || !dims || !data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    if (ndims <= 0 || ndims > 8 || (dtype != MNNR_DTYPE_F32 && dtype != MNNR_DTYPE_I32))
    {
        session->last_error = "invalid dims or dtype";
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::vector<int> shape(dims, dims + ndims);
    for (int d : shape)
    {
        if (d < 0)
        {
            session->last_error = "negative dimension";
            return MNNR_ERROR_INVALID_PARAMETER;
        }
    }

    // Validate against the declared input dtype
    for (size_t i = 0; i < session->input_names.size(); i++)
    {
        if (session->input_names[i] != name)
        {
            continue;
        }
        const halide_type_t &t = session->input_types[i];
        bool declared_int = (t.code == halide_type_int);
        if (declared_int != (dtype == MNNR_DTYPE_I32))
        {
            session->last_error = std::string("dtype mismatch for ") + name +
                                  (declared_int ? ": model expects int32" : ": model expects float32");
            return MNNR_ERROR_INVALID_PARAMETER;
        }
        break;
    }

    int64_t elems = multi_prod(shape);
    std::vector<uint8_t> bytes(static_cast<size_t>(elems) * 4);
    std::memcpy(bytes.data(), data, bytes.size());

    std::lock_guard<std::mutex> lock(session->mutex);
    session->staged_data[name] = std::move(bytes);
    session->staged_dims[name] = std::move(shape);
    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_multi_run(MNN_MultiSession *session)
{
    if (!session)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->staged_data.size() != session->input_names.size())
    {
        session->last_error = "not all inputs staged: expected " +
                              std::to_string(session->input_names.size()) + " got " +
                              std::to_string(session->staged_data.size());
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Build module inputs in module input order (placeholder per input dtype)
    std::vector<MNN::Express::VARP> inputs;
    inputs.reserve(session->input_names.size());
    for (size_t i = 0; i < session->input_names.size(); i++)
    {
        const std::string &name = session->input_names[i];
        const std::vector<int> &dims = session->staged_dims[name];
        const std::vector<uint8_t> &bytes = session->staged_data[name];
        MNN::Express::VARP varp;
        if (session->input_types[i].code == halide_type_int)
        {
            varp = MNN::Express::_Input(dims, MNN::Express::NCHW, halide_type_of<int>());
            std::memcpy(varp->writeMap<int>(), bytes.data(), bytes.size());
        }
        else
        {
            varp = MNN::Express::_Input(dims, MNN::Express::NCHW, halide_type_of<float>());
            std::memcpy(varp->writeMap<float>(), bytes.data(), bytes.size());
        }
        inputs.push_back(std::move(varp));
    }

    auto outputs = session->module->onForward(inputs);
    if (outputs.size() != session->output_names.size())
    {
        session->last_error = "module returned unexpected output count";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    for (size_t i = 0; i < outputs.size(); i++)
    {
        session->last_outputs[session->output_names[i]] = outputs[i];
    }
    return MNNR_SUCCESS;
}

static const MNN::Express::VARP *multi_output_locked(MNN_MultiSession *session, const char *name)
{
    auto it = session->last_outputs.find(name);
    if (it == session->last_outputs.end())
    {
        session->last_error = std::string("output not computed (run first): ") + name;
        return nullptr;
    }
    return &it->second;
}

int64_t mnnr_multi_output_elem_count(MNN_MultiSession *session, const char *name)
{
    if (!session || !name)
    {
        return -MNNR_ERROR_INVALID_PARAMETER;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    const MNN::Express::VARP *varp = multi_output_locked(session, name);
    if (!varp)
    {
        return -MNNR_ERROR_MODEL_LOAD_FAILED;
    }
    const auto *info = (*varp)->getInfo();
    return info ? static_cast<int64_t>(info->size) : -MNNR_ERROR_RUNTIME_ERROR;
}

int32_t mnnr_multi_output_shape(MNN_MultiSession *session, const char *name, int64_t *dims, int32_t cap)
{
    if (!session || !name || !dims || cap <= 0)
    {
        return -MNNR_ERROR_INVALID_PARAMETER;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    const MNN::Express::VARP *varp = multi_output_locked(session, name);
    if (!varp)
    {
        return -MNNR_ERROR_MODEL_LOAD_FAILED;
    }
    const auto *info = (*varp)->getInfo();
    if (!info)
    {
        session->last_error = "output info unavailable";
        return -MNNR_ERROR_RUNTIME_ERROR;
    }
    int32_t ndims = static_cast<int32_t>(info->dim.size());
    if (ndims > cap)
    {
        session->last_error = "shape buffer too small";
        return -MNNR_ERROR_INVALID_PARAMETER;
    }
    for (int32_t i = 0; i < ndims; i++)
    {
        dims[i] = info->dim[i];
    }
    return ndims;
}

MNNR_ErrorCode mnnr_multi_output_f32(MNN_MultiSession *session, const char *name, float *out, int64_t cap_elems)
{
    if (!session || !name || !out)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    const MNN::Express::VARP *varp = multi_output_locked(session, name);
    if (!varp)
    {
        return MNNR_ERROR_MODEL_LOAD_FAILED;
    }
    const auto *info = (*varp)->getInfo();
    if (!info)
    {
        session->last_error = "output info unavailable";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    int64_t elems = static_cast<int64_t>(info->size);
    if (elems > cap_elems)
    {
        session->last_error = "output buffer too small";
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    const float *host = (*varp)->readMap<float>();
    if (!host)
    {
        session->last_error = "output readMap failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    std::memcpy(out, host, static_cast<size_t>(elems) * sizeof(float));
    return MNNR_SUCCESS;
}

const char *mnnr_multi_last_error(MNN_MultiSession *session)
{
    if (!session)
    {
        return "null session";
    }
    return session->last_error.c_str();
}
