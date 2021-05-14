#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_ASSERT

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wdeprecated-volatile"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#pragma clang diagnostic pop

#include <chrono>

#include <vector>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <optional>
#include <set>
#include <cstdint> // Necessary for UINT32_MAX
#include <algorithm> // Necessary for std::min/std::max
#include <fstream>

const int s_MaxFramesInFlight = 2;

const uint32_t s_Width = 1280;
const uint32_t s_Height = 720;

const std::vector<const char*> s_ValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> s_DeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
    const bool s_EnableValidationLayers = false;
#else
    const bool s_EnableValidationLayers = true;
#endif // !NDEBUG

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

struct QueueFamilyIndices {
    std::optional<uint32_t> GraphicsFamily;
    std::optional<uint32_t> PresentFamily;

    bool IsComplete() {
        return GraphicsFamily.has_value() && PresentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR Capabilities;
    std::vector<vk::SurfaceFormatKHR> Formats;
    std::vector<vk::PresentModeKHR> PresentModes;
};

struct Vertex {
    glm::vec2 Position;
    glm::vec3 Color;

    static vk::VertexInputBindingDescription GetBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex,
        };

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> GetAttributeDescriptions() {
        return std::array<vk::VertexInputAttributeDescription, 2>{
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(Vertex, Position),
            },
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, Color),
            },
        };
    }
};

const std::vector<Vertex> s_Vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
};

struct std::vector<uint16_t> s_Indices = {
    0, 1, 2, 2, 3, 0,
};

struct UniformBufferObject {
    alignas(16) glm::mat4 Model;
    alignas(16) glm::mat4 View;
    alignas(16) glm::mat4 Projection;
};

class HelloTriangleApplication {
public:
    bool m_FramebufferResized = false;

private:
    GLFWwindow* m_Window;
    vk::UniqueInstance m_Instance;
    vk::UniqueDebugUtilsMessengerEXT m_DebugMessenger;
    vk::PhysicalDevice m_PhysicalDevice = vk::PhysicalDevice(nullptr);
    vk::UniqueDevice m_Device;
    vk::Queue m_GraphicsQueue;
    vk::Queue m_PresentQueue;
    vk::UniqueSurfaceKHR m_Surface;
    vk::UniqueSwapchainKHR m_OldSwapChain;
    vk::UniqueSwapchainKHR m_SwapChain;
    std::vector<vk::Image> m_SwapChainImages;
    vk::Format m_SwapChainImageFormat;
    vk::Extent2D m_SwapChainExtent;
    std::vector<vk::UniqueImageView> m_SwapChainImageViews;
    vk::UniqueRenderPass m_RenderPass;
    vk::UniqueDescriptorSetLayout m_DescriptorSetLayout;
    vk::UniquePipelineLayout m_PipelineLayout;
    vk::UniquePipeline m_GraphicsPipeline;
    std::vector<vk::UniqueFramebuffer> m_SwapChainFramebuffers;
    vk::UniqueCommandPool m_CommandPool;
    std::vector<vk::UniqueCommandBuffer> m_CommandBuffers;
    std::vector<vk::UniqueSemaphore> m_ImageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> m_RenderFinishedSemaphores;
    std::vector<vk::Fence> m_InFlightFences;
    std::vector<vk::Fence> m_ImagesInFlight;
    size_t m_CurrentFrame = 0;
    vk::UniqueBuffer m_VertexBuffer;
    vk::UniqueDeviceMemory m_VertexBufferMemory;
    vk::UniqueBuffer m_IndexBuffer;
    vk::UniqueDeviceMemory m_IndexBufferMemory;
    std::vector<vk::UniqueBuffer> m_UniformBuffers;
    std::vector<vk::UniqueDeviceMemory> m_UniformBuffersMemory;
    vk::UniqueDescriptorPool m_DescriptorPool;
    std::vector<vk::DescriptorSet> m_DescriptorSets; // Not using unique for now to avoid setting vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet
    vk::UniqueBuffer m_StagingBuffer;
    vk::UniqueDeviceMemory m_StagingBufferMemory;
    vk::UniqueImage m_TextureImage;
    vk::UniqueDeviceMemory m_TextureImageMemory;

public:
    void Run() {
        InitWindow();
        InitVulkan();
        MainLoop();
        Cleanup();
    }

private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL s_DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        VkDebugUtilsMessengerCallbackDataEXT const * pCallbackData,
        void* pUserData) {

        if (messageTypes & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            //|| messageTypes & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
            ) {
            std::cerr << "Validation Layer: " << pCallbackData->pMessage << std::endl;
        }

        return VK_FALSE;
    }

    static void FramebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->m_FramebufferResized = true;
    }

    void InitWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        m_Window = glfwCreateWindow(s_Width, s_Height, "Vulkan", nullptr, nullptr);

        glfwSetWindowUserPointer(m_Window, this);
        glfwSetFramebufferSizeCallback(m_Window, FramebufferResizeCallback);
    }

    void InitVulkan() {
        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateDescriptorSetLayout();
        CreateGraphicsPipeline();
        CreateFramebuffers();
        CreateCommandPool();
        CreateTextureImage();
        CreateVertexBuffer();
        CreateIndexBuffer();
        CreateUniformBuffers();
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
    }

    void CreateInstance() {
        // Setup Dynamic Loader
        vk::DynamicLoader dl;
        PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        if (s_EnableValidationLayers && !CheckValidationLayerSupport()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        vk::ApplicationInfo const appInfo{
            .sType = vk::StructureType::eApplicationInfo,
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(0, 0, 1),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(0, 0, 1),
            .apiVersion = VK_API_VERSION_1_2,
        };

        auto requiredExtensions = GetRequiredExtensions();

        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;

        if (s_EnableValidationLayers) {
            PopulateDebugMessengerCreateInfo(debugCreateInfo);
            /*
            vk::ValidationFeatureEnableEXT enableFeatures[4]{
                vk::ValidationFeatureEnableEXT::eBestPractices,
                vk::ValidationFeatureEnableEXT::eGpuAssisted,
                vk::ValidationFeatureEnableEXT::eGpuAssistedReserveBindingSlot,
                vk::ValidationFeatureEnableEXT::eSynchronizationValidation,
            };

            vk::ValidationFeaturesEXT validationFeatures{
                .sType = vk::StructureType::eValidationFeaturesEXT,
                .enabledValidationFeatureCount = 4,
                .pEnabledValidationFeatures = enableFeatures,
            };

            debugCreateInfo.pNext = &validationFeatures;*/
        }

        vk::InstanceCreateInfo const createInfo {
            .sType = vk::StructureType::eInstanceCreateInfo,
            .pNext = s_EnableValidationLayers ? &debugCreateInfo : nullptr,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = s_EnableValidationLayers ? static_cast<uint32_t>(s_ValidationLayers.size()) : 0,
            .ppEnabledLayerNames = s_EnableValidationLayers ? s_ValidationLayers.data() : nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data(),
        };

        auto [result, instance] = vk::createInstanceUnique(createInfo);

        m_Instance = std::move(instance);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create Vulkan instance!");
        }
        
        VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Instance.get());
    }

    void PopulateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = vk::DebugUtilsMessengerCreateInfoEXT{
            .sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT,
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            .pfnUserCallback = s_DebugCallback,
        };
    }

    void SetupDebugMessenger() {
        if (!s_EnableValidationLayers) return;

        vk::DebugUtilsMessengerCreateInfoEXT createInfo;
        PopulateDebugMessengerCreateInfo(createInfo);

        auto [result, debugMessenger] = m_Instance->createDebugUtilsMessengerEXTUnique(createInfo, nullptr);

        m_DebugMessenger = std::move(debugMessenger);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to setup debug messenger!");
        }
    }

    std::vector<const char*> GetRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (s_EnableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool CheckValidationLayerSupport() {
        auto [result, availableLayers] = vk::enumerateInstanceLayerProperties();

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to enumerate instance layer properties!");
        }

        for (const auto layerName : s_ValidationLayers)
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if((std::string_view)layerProperties.layerName == (std::string_view)layerName)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    bool CheckDeviceExtensionSupport(vk::PhysicalDevice device) {
        auto [result, availableExtensions] = device.enumerateDeviceExtensionProperties();

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to enumerate device extension properties!");
        }

        std::set<std::string> requiredExtensions(s_DeviceExtensions.begin(), s_DeviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    bool IsDeviceSuitable(vk::PhysicalDevice device) {
        QueueFamilyIndices indices = FindQueueFamilies(device);
        bool extensionsSupported = CheckDeviceExtensionSupport(device);
        bool swapChainAdequate = false;

        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.Formats.empty() && !swapChainSupport.PresentModes.empty();
        }

        return indices.IsComplete() && extensionsSupported && swapChainAdequate;
    }

    void PickPhysicalDevice() {
        auto [result, devices] = m_Instance->enumeratePhysicalDevices();

        if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to enumerate physical devices!");
        }

        if (devices.empty())
        {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        for (const auto& device : devices) {
            if (IsDeviceSuitable(device))
            {
                m_PhysicalDevice = device;
                break;
            }
        }

        if (m_PhysicalDevice == vk::PhysicalDevice(nullptr)) {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }

    QueueFamilyIndices FindQueueFamilies(vk::PhysicalDevice device) {
        QueueFamilyIndices indices;

        auto queueFamilies = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.GraphicsFamily = i;
            }

            auto [result, presentSupport] = device.getSurfaceSupportKHR(i, m_Surface.get());

            if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to get surface support!");
            }

            if (presentSupport) {
                indices.PresentFamily = i;
            }

            if (indices.IsComplete())
            {
                break;
            }

            i++;
        }

        return indices;
    }

    void CreateLogicalDevice() {
        QueueFamilyIndices indices = FindQueueFamilies(m_PhysicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> unqueQueueFamilies = { indices.GraphicsFamily.value(), indices.PresentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : unqueQueueFamilies) {
            queueCreateInfos.push_back(vk::DeviceQueueCreateInfo{
                .sType = vk::StructureType::eDeviceQueueCreateInfo,
                .queueFamilyIndex = queueFamily,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority,
            });
        }
        
        vk::PhysicalDeviceFeatures deviceFeatures{};

        vk::DeviceCreateInfo createInfo{
            .sType = vk::StructureType::eDeviceCreateInfo,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledLayerCount = s_EnableValidationLayers ? static_cast<uint32_t>(s_ValidationLayers.size()) : 0,
            .ppEnabledLayerNames = s_EnableValidationLayers ? s_ValidationLayers.data() : nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(s_DeviceExtensions.size()),
            .ppEnabledExtensionNames = s_DeviceExtensions.data(),
            .pEnabledFeatures = &deviceFeatures,
        };

        auto [result, device] = m_PhysicalDevice.createDeviceUnique(createInfo);

        m_Device = std::move(device);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create logical device!");
        }

        VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Device.get());

        m_GraphicsQueue = m_Device->getQueue(indices.GraphicsFamily.value(), 0);
        m_PresentQueue = m_Device->getQueue(indices.PresentFamily.value(), 0);
    }

    void CreateSurface() {
        VkSurfaceKHR surface;
        vk::Result result = (vk::Result)glfwCreateWindowSurface(m_Instance.get(), m_Window, nullptr, &surface);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create window surface!");
        }

        m_Surface = vk::UniqueSurfaceKHR(surface, m_Instance.get());
    }

    SwapChainSupportDetails QuerySwapChainSupport(vk::PhysicalDevice device) {
        SwapChainSupportDetails details;

        auto [capabilitiesResult, capabilities] = device.getSurfaceCapabilitiesKHR(m_Surface.get());

        details.Capabilities = capabilities;

        if ( capabilitiesResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to get surface capabilities");
        }

        auto [formatResult, format] = device.getSurfaceFormatsKHR(m_Surface.get());

        details.Formats = format;

        if (formatResult != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to get surface format!");
        }

        auto [presentModesResult, presentModes] = device.getSurfacePresentModesKHR(m_Surface.get());

        details.PresentModes = presentModes;

        if (presentModesResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to get surface present modes!");
        }

        return details;
    }

    vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && 
                availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if(capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(m_Window, &width, &height);

            vk::Extent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
            
        }
    }

    void CreateSwapChain() {
        SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(m_PhysicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.Formats);
        vk::PresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.PresentModes);
        vk::Extent2D extent = ChooseSwapExtent(swapChainSupport.Capabilities);

        uint32_t imageCount = swapChainSupport.Capabilities.minImageCount + 1;
        if (swapChainSupport.Capabilities.maxImageCount > 0 && imageCount > swapChainSupport.Capabilities.maxImageCount) {
            imageCount = swapChainSupport.Capabilities.maxImageCount;
        }

        QueueFamilyIndices indices = FindQueueFamilies(m_PhysicalDevice);
        uint32_t queueFamilyIndices[] = { indices.GraphicsFamily.value(), indices.PresentFamily.value() };

        bool seperateQueueFamilies = indices.GraphicsFamily != indices.PresentFamily;

        m_OldSwapChain = std::move(m_SwapChain);

        vk::SwapchainCreateInfoKHR createInfo{
            .sType = vk::StructureType::eSwapchainCreateInfoKHR,
            .surface = m_Surface.get(),
            .minImageCount = imageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = seperateQueueFamilies ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = seperateQueueFamilies ? (uint32_t)2 : (uint32_t)0,
            .pQueueFamilyIndices = seperateQueueFamilies ? queueFamilyIndices : nullptr,
            .preTransform = swapChainSupport.Capabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = m_OldSwapChain.get(),
        };

        auto [result, swapChain] = m_Device->createSwapchainKHRUnique(createInfo);

        m_SwapChain = std::move(swapChain);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create swap chain!");
        }

        auto [swapChainImagesResult, swapChainImages] = m_Device->getSwapchainImagesKHR(m_SwapChain.get());

        m_SwapChainImages = std::move(swapChainImages);

        if (swapChainImagesResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to get swapchain images!");
        }

        m_SwapChainImageFormat = surfaceFormat.format;
        m_SwapChainExtent = extent;
    }

    void CleanupSwapChain() {
        m_SwapChainFramebuffers.clear();
        m_CommandBuffers.clear();
        m_RenderPass.release();
        m_SwapChainImageViews.clear();

        // Not sure why this is needed to prevent validation error / crashes, the next line calls the destroy function
        m_Device->destroySwapchainKHR(m_OldSwapChain.get());
        m_OldSwapChain.release();

        m_UniformBuffers.clear();
        m_UniformBuffersMemory.clear();

        m_DescriptorPool.release();
    }

    void RecreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(m_Window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(m_Window, &width, &height);
            glfwWaitEvents();
        }

        if(m_Device->waitIdle() != vk::Result::eSuccess)
        {
            throw std::runtime_error("Wait idle failed!");
        }

        CleanupSwapChain();

        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateFramebuffers();
        CreateUniformBuffers();
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateCommandBuffers();

        m_ImagesInFlight.resize(m_SwapChainImages.size(), vk::Fence(nullptr));
    }

    void CreateImageViews() {
        m_SwapChainImageViews.resize(m_SwapChainImages.size());

        for (size_t i = 0; i < m_SwapChainImages.size(); i++) {
            vk::ImageViewCreateInfo createInfo{
                .sType = vk::StructureType::eImageViewCreateInfo,
                .image = m_SwapChainImages[i],
                .viewType = vk::ImageViewType::e2D,
                .format = m_SwapChainImageFormat,
                .components = {
                    .r = vk::ComponentSwizzle::eIdentity,
                    .g = vk::ComponentSwizzle::eIdentity,
                    .b = vk::ComponentSwizzle::eIdentity,
                    .a = vk::ComponentSwizzle::eIdentity,
                },
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };

            auto [result, imageView] = m_Device->createImageViewUnique(createInfo);

            m_SwapChainImageViews[i] = std::move(imageView);

            if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to create image views!");
            }
        }
    }

    void CreateRenderPass() {
        vk::AttachmentDescription colorAttachment{
            .format = m_SwapChainImageFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::ePresentSrcKHR,
        };

        vk::AttachmentReference colorAttachmentRef{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
        };

        vk::SubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eNoneKHR,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        };

        vk::RenderPassCreateInfo renderPassInfo{
            .sType = vk::StructureType::eRenderPassCreateInfo,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        auto [result, renderPass] = m_Device->createRenderPassUnique(renderPassInfo);

        m_RenderPass = std::move(renderPass);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create render pass!");
        }
    }

    vk::ShaderModule CreateShaderModule(const std::vector<char>& code) {
        vk::ShaderModuleCreateInfo createInfo{
            .sType = vk::StructureType::eShaderModuleCreateInfo,
            .codeSize = code.size(),
            .pCode = reinterpret_cast<const uint32_t*>(code.data()),
        };

        auto [result, shaderModule] = m_Device->createShaderModule(createInfo);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create shader module!");
        }

        return shaderModule;
    }

    void CreateDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
            .bindingCount = 1,
            .pBindings = &uboLayoutBinding,
        };

        auto [result, descriptorSetLayout] = m_Device->createDescriptorSetLayoutUnique(layoutInfo);

        m_DescriptorSetLayout = std::move(descriptorSetLayout);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }

    void CreateGraphicsPipeline() {
        auto vertShaderCode = ReadFile("shaders/shader.vert.spv");
        auto fragShaderCode = ReadFile("shaders/shader.frag.spv");

        vk::ShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
        vk::ShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertShaderModule,
            .pName = "main",
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragShaderModule,
            .pName = "main",
        };

        vk::PipelineShaderStageCreateInfo shaderStages[]{vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::GetBindingDescription();
        auto attributeDescriptions = Vertex::GetAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data(),
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
            .sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo,
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE,
        };

        vk::PipelineViewportStateCreateInfo viewportStateInfo{
            .sType = vk::StructureType::ePipelineViewportStateCreateInfo,
            .viewportCount = 1,
            .pViewports = nullptr, // Dynamic state, so null
            .scissorCount = 1,
            .pScissors = nullptr, // Dynamic state, so null
        };

        vk::PipelineRasterizationStateCreateInfo rasterizerInfo{
            .sType = vk::StructureType::ePipelineRasterizationStateCreateInfo,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eClockwise,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };

        vk::PipelineMultisampleStateCreateInfo multisamplingInfo{
            .sType = vk::StructureType::ePipelineMultisampleStateCreateInfo,
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE,
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | 
                                vk::ColorComponentFlagBits::eG | 
                                vk::ColorComponentFlagBits::eB | 
                                vk::ColorComponentFlagBits::eA,
        };

        vk::PipelineColorBlendStateCreateInfo colorBlendingInfo{
            .sType = vk::StructureType::ePipelineColorBlendStateCreateInfo,
            .logicOpEnable = VK_FALSE,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };

        colorBlendingInfo.blendConstants[0] = 0.0f;
        colorBlendingInfo.blendConstants[1] = 0.0f;
        colorBlendingInfo.blendConstants[2] = 0.0f;
        colorBlendingInfo.blendConstants[3] = 0.0f;

        

        vk::DynamicState dynamicStates[]{
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
            .sType = vk::StructureType::ePipelineDynamicStateCreateInfo,
            .dynamicStateCount = 2,
            .pDynamicStates = dynamicStates,
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = vk::StructureType::ePipelineLayoutCreateInfo,
            .setLayoutCount = 1,
            .pSetLayouts = &m_DescriptorSetLayout.get(),
        };

        auto [result, pipelineLayout] = m_Device->createPipelineLayoutUnique(pipelineLayoutInfo);

        m_PipelineLayout = std::move(pipelineLayout);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportStateInfo,
            .pRasterizationState = &rasterizerInfo,
            .pMultisampleState = &multisamplingInfo,
            .pColorBlendState = &colorBlendingInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = m_PipelineLayout.get(),
            .renderPass = m_RenderPass.get(),
            .subpass = 0,
            .basePipelineHandle = vk::Pipeline(nullptr),
        };

        auto [pipelineResult, graphicsPipeline] = m_Device->createGraphicsPipelineUnique(vk::PipelineCache(nullptr), pipelineInfo);

        m_GraphicsPipeline = std::move(graphicsPipeline);

        if (pipelineResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create graphics pipeline!");
        }

        m_Device->destroyShaderModule(vertShaderModule);
        m_Device->destroyShaderModule(fragShaderModule);
    }

    void CreateFramebuffers() {
        m_SwapChainFramebuffers.resize(m_SwapChainImageViews.size());

        for (size_t i = 0; i < m_SwapChainImageViews.size(); i++) {
            vk::ImageView attachments[] = {
                m_SwapChainImageViews[i].get(),
            };

            vk::FramebufferCreateInfo framebufferInfo{
                .sType = vk::StructureType::eFramebufferCreateInfo,
                .renderPass = m_RenderPass.get(),
                .attachmentCount = 1,
                .pAttachments = attachments,
                .width = m_SwapChainExtent.width,
                .height = m_SwapChainExtent.height,
                .layers = 1,
            };

            auto [result, framebuffer] = m_Device->createFramebufferUnique(framebufferInfo);

            m_SwapChainFramebuffers[i] = std::move(framebuffer);

            if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }
    }

    void CreateCommandPool() {
        QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(m_PhysicalDevice);

        vk::CommandPoolCreateInfo poolInfo{
            .sType = vk::StructureType::eCommandPoolCreateInfo,
            .queueFamilyIndex = queueFamilyIndices.GraphicsFamily.value(),
        };

        auto [result, commandPool] = m_Device->createCommandPoolUnique(poolInfo);

        m_CommandPool = std::move(commandPool);

        if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to create command pool!");
        }
    }

    void CreateImageUnique(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
        vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::UniqueImage& image, 
        vk::UniqueDeviceMemory& imageMemory) {

        vk::Image localImage;
        vk::DeviceMemory localImageMemory;
        CreateImage(width, height, format, tiling, usage, properties, localImage, localImageMemory);

        image = vk::UniqueImage(localImage, m_Device.get());
        imageMemory = vk::UniqueDeviceMemory(localImageMemory, m_Device.get());
    }

    void CreateImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
                        vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, 
                        vk::DeviceMemory& imageMemory) {
    
        vk::ImageCreateInfo imageInfo{
            .sType = vk::StructureType::eImageCreateInfo,
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {
                .width = static_cast<uint32_t>(width),
                .height = static_cast<uint32_t>(height),
                .depth = 1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };

        auto [result, localImage] = m_Device->createImage(imageInfo);

        image = std::move(localImage);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to allocate image memory!");
        }

        vk::MemoryRequirements memRequirements = m_Device->getImageMemoryRequirements(image);

        vk::MemoryAllocateInfo allocInfo{
            .sType = vk::StructureType::eMemoryAllocateInfo,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties),
        };

        auto [localImageMemoryResult, localImageMemory] = m_Device->allocateMemory(allocInfo);

        imageMemory = std::move(localImageMemory);

        if (localImageMemoryResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to allocate image memory!");
        }

        if (m_Device->bindImageMemory(image, imageMemory, 0) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to bind image memory!");
        }
    }

    void CreateTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("Failed to load texture image!");
        }

        CreateBufferUnique(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            m_StagingBuffer, m_StagingBufferMemory);

        void* data;
        if (m_Device->mapMemory(m_StagingBufferMemory.get(), 0, imageSize, vk::MemoryMapFlags(0), &data) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to map memory!");
        }
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        m_Device->unmapMemory(m_StagingBufferMemory.get());

        stbi_image_free(pixels);

        CreateImageUnique(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, m_TextureImage, m_TextureImageMemory);

        TransitionImageLayout(m_TextureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        CopyBufferToImage(m_StagingBuffer.get(), m_TextureImage.get(), static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        TransitionImageLayout(m_TextureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        m_StagingBuffer.release();
        m_StagingBufferMemory.release();
    }

    void TransitionImageLayout(const vk::UniqueImage& image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::UniqueCommandBuffer commandBuffer = BeginSingleTimeCommands();

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        vk::ImageMemoryBarrier barrier{
            .sType = vk::StructureType::eImageMemoryBarrier,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.get(),
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        if (oldLayout == vk::ImageLayout::eUndefined && 
            newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && 
                    newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("Unsupported layout transition!");
        }

        commandBuffer->pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags(0), nullptr, nullptr, barrier);

        EndSingleTimeCommands(commandBuffer);
    }

    void CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        vk::UniqueCommandBuffer commandBuffer = BeginSingleTimeCommands();

        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = { 
                .x = 0,
                .y = 0,
                .z = 0,
            },
            .imageExtent = {
                .width = width,
                .height = height,
                .depth = 1,
            },
        };

        commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);

        EndSingleTimeCommands(commandBuffer);
    }

    uint32_t FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = m_PhysicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }

    void CreateBufferUnique(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                        vk::UniqueBuffer& buffer, vk::UniqueDeviceMemory& bufferMemory) {
        vk::Buffer localBuffer;
        vk::DeviceMemory localBufferMemory;
        CreateBuffer(size, usage, properties, localBuffer, localBufferMemory);
        
        buffer = vk::UniqueBuffer(localBuffer, m_Device.get());
        bufferMemory = vk::UniqueDeviceMemory(localBufferMemory, m_Device.get());
    }

    void CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
        vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
        vk::BufferCreateInfo bufferInfo{
            .sType = vk::StructureType::eBufferCreateInfo,
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        auto [result, localBuffer] = m_Device->createBuffer(bufferInfo);

        buffer = localBuffer;

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create vertex buffer!");
        }

        vk::MemoryRequirements memRequirements = m_Device->getBufferMemoryRequirements(buffer);

        vk::MemoryAllocateInfo allocInfo{
            .sType = vk::StructureType::eMemoryAllocateInfo,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties),
        };

        {
            auto [result, localBufferMemory] = m_Device->allocateMemory(allocInfo);

            bufferMemory = localBufferMemory;

            if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to allocate vertex buffer memory!");
            }
        }

        if (m_Device->bindBufferMemory(buffer, bufferMemory, 0) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to bind buffer memory!");
        }
    }

    void CopyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
        vk::UniqueCommandBuffer commandBuffer = BeginSingleTimeCommands();

        vk::BufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size,
        };
        commandBuffer->copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

        EndSingleTimeCommands(commandBuffer);
    }

    template<typename t>
    void CreateVKBufferUnique(const std::vector<t>& data, vk::UniqueBuffer& buffer,
                                vk::UniqueDeviceMemory& bufferMemory, vk::BufferUsageFlags usage)
    {
        vk::Buffer localBuffer;
        vk::DeviceMemory localBufferMemory;
        CreateVKBuffer<t>(data, localBuffer, localBufferMemory, usage);

        buffer = vk::UniqueBuffer(localBuffer, m_Device.get());
        bufferMemory = vk::UniqueDeviceMemory(localBufferMemory, m_Device.get());
    }

    template<typename t>
    void CreateVKBuffer(const std::vector<t>& data, vk::Buffer& buffer,
                        vk::DeviceMemory& bufferMemory, vk::BufferUsageFlags usage) {
        vk::DeviceSize bufferSize = sizeof(data[0]) * data.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        CreateBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory);

        void* memoryData;
        if (m_Device->mapMemory(stagingBufferMemory, vk::DeviceSize(0), 
            bufferSize, vk::MemoryMapFlags(0), &memoryData) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to map memory!");
        }
        memcpy(memoryData, data.data(), (size_t)bufferSize);
        m_Device->unmapMemory(stagingBufferMemory);

        CreateBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | usage,
            vk::MemoryPropertyFlagBits::eDeviceLocal, buffer, bufferMemory);

        CopyBuffer(stagingBuffer, buffer, bufferSize);

        m_Device->destroyBuffer(stagingBuffer);
        m_Device->freeMemory(stagingBufferMemory);
    }

    void CreateVertexBuffer() {
        CreateVKBufferUnique<Vertex>(s_Vertices, m_VertexBuffer, m_VertexBufferMemory, vk::BufferUsageFlagBits::eVertexBuffer);
    }

    void CreateIndexBuffer() {
        CreateVKBufferUnique<uint16_t>(s_Indices, m_IndexBuffer, m_IndexBufferMemory, vk::BufferUsageFlagBits::eIndexBuffer);
    }

    void CreateUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        m_UniformBuffers.resize(m_SwapChainImages.size());
        m_UniformBuffersMemory.resize(m_SwapChainImages.size());

        for (size_t i = 0; i < m_SwapChainImages.size(); i++) {
            CreateBufferUnique(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                m_UniformBuffers[i], m_UniformBuffersMemory[i]);
        }
    }

    void CreateDescriptorPool() {
        vk::DescriptorPoolSize poolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = static_cast<uint32_t>(m_SwapChainImages.size()),
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .sType = vk::StructureType::eDescriptorPoolCreateInfo,
            .maxSets = static_cast<uint32_t>(m_SwapChainImages.size()),
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        };

        auto [result, descriptorPool] = m_Device->createDescriptorPoolUnique(poolInfo);

        m_DescriptorPool = std::move(descriptorPool);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create descriptor pool!");
        }
    }

    void CreateDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(m_SwapChainImages.size(), m_DescriptorSetLayout.get());

        vk::DescriptorSetAllocateInfo allocInfo{
            .sType = vk::StructureType::eDescriptorSetAllocateInfo,
            .descriptorPool = m_DescriptorPool.get(),
            .descriptorSetCount = static_cast<uint32_t>(m_SwapChainImages.size()),
            .pSetLayouts = layouts.data(),
        };

        auto [result, descriptorSets] = m_Device->allocateDescriptorSets(allocInfo);

        m_DescriptorSets = std::move(descriptorSets);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < m_SwapChainImages.size(); i++) {
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = m_UniformBuffers[i].get(),
                .offset = 0,
                .range = sizeof(UniformBufferObject),
            };

            vk::WriteDescriptorSet descriptorWrite{
                .sType = vk::StructureType::eWriteDescriptorSet,
                .dstSet = m_DescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo,
            };

            m_Device->updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
        }
    }

    void CreateCommandBuffers(){
        vk::CommandBufferAllocateInfo allocInfo{
            .sType = vk::StructureType::eCommandBufferAllocateInfo,
            .commandPool = m_CommandPool.get(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(m_SwapChainFramebuffers.size()),
        };

        vk::Viewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = (float)m_SwapChainExtent.width,
            .height = (float)m_SwapChainExtent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = m_SwapChainExtent,
        };

        auto [allocResult, commandBuffers] = m_Device->allocateCommandBuffersUnique(allocInfo);

        m_CommandBuffers = std::move(commandBuffers);

        if (allocResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }

        for (size_t i = 0; i < m_CommandBuffers.size(); i++) {
            vk::CommandBufferBeginInfo beginInfo{
                .sType = vk::StructureType::eCommandBufferBeginInfo,
            };

            if (m_CommandBuffers[i]->begin(beginInfo) != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to begin recording command buffer!");
            }

            vk::ClearValue clearValue{};
            clearValue.color = vk::ClearColorValue(std::array<float,4>{ 0.0f, 0.0f, 0.0f, 1.0f });

            vk::RenderPassBeginInfo renderPassInfo{
                .sType = vk::StructureType::eRenderPassBeginInfo,
                .renderPass = m_RenderPass.get(),
                .framebuffer = m_SwapChainFramebuffers[i].get(),
                .renderArea = {
                    .offset = {0, 0},
                    .extent = m_SwapChainExtent,
                },
                .clearValueCount = 1,
                .pClearValues = &clearValue,
            };

            m_CommandBuffers[i]->setViewport(0, 1, &viewport);
            m_CommandBuffers[i]->setScissor(0, 1, &scissor);

            m_CommandBuffers[i]->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

            m_CommandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, m_GraphicsPipeline.get());

            vk::Buffer vertexBuffers[] = { m_VertexBuffer.get() };
            vk::DeviceSize offsets[] = { 0 };
            m_CommandBuffers[i]->bindVertexBuffers(0, 1, vertexBuffers, offsets);

            m_CommandBuffers[i]->bindIndexBuffer(m_IndexBuffer.get(), 0, vk::IndexType::eUint16);

            m_CommandBuffers[i]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_PipelineLayout.get(), 0, 1, &m_DescriptorSets[i], 0, nullptr);

            m_CommandBuffers[i]->drawIndexed(static_cast<uint32_t>(s_Indices.size()), 1, 0, 0, 0);

            m_CommandBuffers[i]->endRenderPass();

            if (m_CommandBuffers[i]->end() != vk::Result::eSuccess)
            {
                throw std::runtime_error("Failed to record command buffer!");
            }
        }
    }

    void CreateSyncObjects() {
        m_ImageAvailableSemaphores.resize(s_MaxFramesInFlight);
        m_RenderFinishedSemaphores.resize(s_MaxFramesInFlight);
        m_InFlightFences.resize(s_MaxFramesInFlight);
        m_ImagesInFlight.resize(m_SwapChainImages.size(), vk::Fence(nullptr));

        vk::SemaphoreCreateInfo semaphoreInfo{
            .sType = vk::StructureType::eSemaphoreCreateInfo,
        };

        vk::FenceCreateInfo fenceInfo{
            .sType = vk::StructureType::eFenceCreateInfo,
            .flags = vk::FenceCreateFlagBits::eSignaled,
        };

        for (size_t i = 0; i < s_MaxFramesInFlight; i++)
        {
            auto [imageAvailableResult, imageAvailableSemaphore] = m_Device->createSemaphoreUnique(semaphoreInfo);
            auto [renderFinishedResult, renderFinishedSemaphore] = m_Device->createSemaphoreUnique(semaphoreInfo);
            auto [inFlightFenceResult, inFlightFence] = m_Device->createFence(fenceInfo);

            m_ImageAvailableSemaphores[i] = std::move(imageAvailableSemaphore);
            m_RenderFinishedSemaphores[i] = std::move(renderFinishedSemaphore);
            m_InFlightFences[i] = std::move(inFlightFence);

            if (imageAvailableResult != vk::Result::eSuccess || 
                renderFinishedResult != vk::Result::eSuccess || 
                inFlightFenceResult != vk::Result::eSuccess)
            {
                throw std::runtime_error("Failed to create semaphores!");
            }
        }
    }

    void UpdateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{
            .Model = glm::rotate(glm::mat4(1.0f), time * glm::half_pi<float>(), glm::vec3(0.0f, 0.0f, 1.0f)),
            .View = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
            .Projection = glm::perspective(glm::radians(45.0f), (float)m_SwapChainExtent.width / (float)m_SwapChainExtent.height, 0.1f, 10.0f),
        };
        ubo.Projection[1][1] *= -1.0f; // Y Flip (glm uses OpenGL coords)

        void* data;
        if (m_Device->mapMemory(m_UniformBuffersMemory[currentImage].get(), vk::DeviceSize(0), 
            sizeof(ubo), vk::MemoryMapFlags(0), &data) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to map memory!");
        }
        memcpy(data, &ubo, sizeof(ubo));
        m_Device->unmapMemory(m_UniformBuffersMemory[currentImage].get());
    }

    void DrawFrame() {
        if (m_Device->waitForFences(m_InFlightFences[m_CurrentFrame], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to wait for fence!");
        }

        auto [result, imageIndex] = m_Device->acquireNextImageKHR(m_SwapChain.get(), UINT64_MAX, m_ImageAvailableSemaphores[m_CurrentFrame].get(), vk::Fence(nullptr));

        if (result == vk::Result::eErrorOutOfDateKHR) {
            RecreateSwapChain();
            return;
        } else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (m_ImagesInFlight[imageIndex] != vk::Fence(nullptr)) {
            if (m_Device->waitForFences(m_ImagesInFlight[imageIndex], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to wait for fence!");
            }
        }

        m_ImagesInFlight[imageIndex] = m_InFlightFences[m_CurrentFrame];

        UpdateUniformBuffer(imageIndex);

        vk::Semaphore waitSemaphores[] = { m_ImageAvailableSemaphores[m_CurrentFrame].get() };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::Semaphore signalSemaphores[] = { m_RenderFinishedSemaphores[m_CurrentFrame].get() };

        vk::SubmitInfo submitInfo{
            .sType = vk::StructureType::eSubmitInfo,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = waitSemaphores,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &m_CommandBuffers[imageIndex].get(),
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = signalSemaphores,
        };

        if (m_Device->resetFences(m_InFlightFences[m_CurrentFrame]) != vk::Result::eSuccess) {
            std::runtime_error("Reset fences failed!");
        }

        if (m_GraphicsQueue.submit(1, &submitInfo, m_InFlightFences[m_CurrentFrame]) != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        vk::SwapchainKHR swapChains[] = { m_SwapChain.get() };

        vk::PresentInfoKHR presentInfo{
            .sType = vk::StructureType::ePresentInfoKHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = signalSemaphores,
            .swapchainCount = 1,
            .pSwapchains = swapChains,
            .pImageIndices = &imageIndex,
        };

        result = m_PresentQueue.presentKHR(presentInfo);

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || m_FramebufferResized) {
            m_FramebufferResized = false;
            RecreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present swap chain image!");
        }

        m_CurrentFrame = (m_CurrentFrame + 1) % s_MaxFramesInFlight;
    }

    void MainLoop() {
        while (!glfwWindowShouldClose(m_Window)) {
            glfwPollEvents();
            DrawFrame();
        }

        if(m_Device->waitIdle() != vk::Result::eSuccess)
        {
            throw std::runtime_error("Wait idle failed!");
        }
    }

    void Cleanup() {
        m_Device.release();
        m_Surface.release();

        if (s_EnableValidationLayers)
        {
            m_DebugMessenger.release();
        }

        m_Instance.release();

        glfwDestroyWindow(m_Window);

        glfwTerminate();
    }

    static std::vector<char> ReadFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    vk::UniqueCommandBuffer BeginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo{
            .sType = vk::StructureType::eCommandBufferAllocateInfo,
            .commandPool = m_CommandPool.get(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        auto [result, commandBuffers] = m_Device->allocateCommandBuffersUnique(allocInfo);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }

        vk::CommandBufferBeginInfo beginInfo{
            .sType = vk::StructureType::eCommandBufferBeginInfo,
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };
        if (commandBuffers[0]->begin(beginInfo) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }

        return std::move(commandBuffers[0]);
    }

    void EndSingleTimeCommands(const vk::UniqueCommandBuffer& commandBuffer) {
        if (commandBuffer->end() != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to record command buffer!");
        }

        vk::SubmitInfo submitInfo{
            .sType = vk::StructureType::eSubmitInfo,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer.get(),
        };

        if (m_GraphicsQueue.submit(1, &submitInfo, vk::Fence(nullptr)) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to submit copy command buffer!");
        }

        if (m_GraphicsQueue.waitIdle() != vk::Result::eSuccess) {
            throw std::runtime_error("Wait idle failed!");
        }
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.Run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}