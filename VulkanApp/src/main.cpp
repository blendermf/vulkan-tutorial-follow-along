#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_ASSERT 
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

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
        CreateGraphicsPipeline();
        CreateFramebuffers();
        CreateCommandPool();
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
        m_GraphicsPipeline.release();
        m_PipelineLayout.release();
        m_RenderPass.release();
        m_SwapChainImageViews.clear();

        // Not sure why this is needed to prevent validation error / crashes, the next line calls the destroy function
        m_Device->destroySwapchainKHR(m_OldSwapChain.get());
        m_OldSwapChain.release();
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
        CreateGraphicsPipeline();
        CreateFramebuffers();
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

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
            .vertexBindingDescriptionCount = 0,
            .vertexAttributeDescriptionCount = 0,
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
            .sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo,
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE,
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

        vk::PipelineViewportStateCreateInfo viewportStateInfo{
            .sType = vk::StructureType::ePipelineViewportStateCreateInfo,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
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

        /*
        // not currently using
        vk::DynamicState dynamicStates[]{
            vk::DynamicState::eViewport,
            vk::DynamicState::eLineWidth
        };

        vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
            .sType = vk::StructureType::ePipelineDynamicStateCreateInfo,
            .dynamicStateCount = 2,
            .pDynamicStates = dynamicStates,
        };
        */

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = vk::StructureType::ePipelineLayoutCreateInfo,
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0,
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
            //.pDynamicState = nullptr,
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

    void CreateCommandBuffers(){
        vk::CommandBufferAllocateInfo allocInfo{
            .sType = vk::StructureType::eCommandBufferAllocateInfo,
            .commandPool = m_CommandPool.get(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(m_SwapChainFramebuffers.size()),
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

            m_CommandBuffers[i]->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

            m_CommandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, m_GraphicsPipeline.get());

            m_CommandBuffers[i]->draw(3, 1, 0, 0);

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