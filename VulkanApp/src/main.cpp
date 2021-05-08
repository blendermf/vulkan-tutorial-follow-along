#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
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

class HelloTriangleApplication {
private:
    GLFWwindow* m_Window;
    vk::UniqueInstance m_Instance;
    vk::UniqueDebugUtilsMessengerEXT m_DebugMessenger;
    vk::PhysicalDevice m_PhysicalDevice = vk::PhysicalDevice();
    vk::UniqueDevice m_Device;
    vk::Queue m_GraphicsQueue;
    vk::Queue m_PresentQueue;
    vk::UniqueSurfaceKHR m_Surface;
    vk::UniqueSwapchainKHR m_SwapChain;
    std::vector<vk::Image> m_SwapChainImages;
    vk::Format m_SwapChainImageFormat;
    vk::Extent2D m_SwapChainExtent;
    std::vector<vk::UniqueImageView> m_SwapChainImageViews;
    vk::UniqueRenderPass m_RenderPass;
    vk::UniquePipelineLayout m_PipelineLayout;

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

        if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            std::cerr << "Validation Layer: " << pCallbackData->pMessage << std::endl;
        }

        return VK_FALSE;
    }

    void InitWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_Window = glfwCreateWindow(s_Width, s_Height, "Vulkan", nullptr, nullptr);
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

        /*
        // Display available Vulkan extensions
        auto availableExtensions = vk::enumerateInstanceExtensionProperties().value;

        std::cout << "Available Vulkan Extensions:\n";
        for (const auto& extension : availableExtensions) {
            std::cout << '\t' << extension.extensionName << '\n';
        }
        */

        auto requiredExtensions = getRequiredExtensions();

        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;

        if (s_EnableValidationLayers) {
            PopulateDebugMessengerCreateInfo(debugCreateInfo);
        }

        vk::InstanceCreateInfo const createInfo {
            .sType = vk::StructureType::eInstanceCreateInfo,
            .pNext = s_EnableValidationLayers ? (vk::DebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo : nullptr,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = s_EnableValidationLayers ? static_cast<uint32_t>(s_ValidationLayers.size()) : 0,
            .ppEnabledLayerNames = s_EnableValidationLayers ? s_ValidationLayers.data() : nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data(),
        };

        auto [result, instance] = vk::createInstanceUnique(createInfo, nullptr);

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
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
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
    }

    std::vector<const char*> getRequiredExtensions() {
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
        auto availableLayers = vk::enumerateInstanceLayerProperties().value;

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
        auto availableExtensions = device.enumerateDeviceExtensionProperties().value;

        std::set<std::string> requiredExtensions(s_DeviceExtensions.begin(), s_DeviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    bool IsDeviceSuitable(vk::PhysicalDevice device) {
        vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
        vk::PhysicalDeviceFeatures deviceFeatures = device.getFeatures();

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
        auto devices = m_Instance->enumeratePhysicalDevices().value;

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

        if (m_PhysicalDevice == VK_NULL_HANDLE) {
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

                auto [result, presentSupport] = device.getSurfaceSupportKHR(i, m_Surface.get());

                if (presentSupport) {
                    indices.PresentFamily = i;
                }
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

        float queuePriority = 1.0f;

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> unqueQueueFamilies = { indices.GraphicsFamily.value(), indices.PresentFamily.value() };

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

        details.Capabilities = device.getSurfaceCapabilitiesKHR(m_Surface.get()).value;
        details.Formats = device.getSurfaceFormatsKHR(m_Surface.get()).value;
        details.PresentModes = device.getSurfacePresentModesKHR(m_Surface.get()).value;

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

        m_SwapChainImageFormat = surfaceFormat.format;
        m_SwapChainExtent = extent;

        uint32_t imageCount = swapChainSupport.Capabilities.minImageCount + 1;

        if (swapChainSupport.Capabilities.maxImageCount > 0 && imageCount > swapChainSupport.Capabilities.maxImageCount) {
            imageCount = swapChainSupport.Capabilities.maxImageCount;
        }

        QueueFamilyIndices indices = FindQueueFamilies(m_PhysicalDevice);
        uint32_t queueFamilyIndices[] = { indices.GraphicsFamily.value(), indices.PresentFamily.value() };

        bool seperateQueueFamilies = indices.GraphicsFamily != indices.PresentFamily;

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
            .oldSwapchain = nullptr,
        };

        auto [result, swapChain] = m_Device->createSwapchainKHRUnique(createInfo);

        m_SwapChain = std::move(swapChain);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create swap chain!");
        }

        m_SwapChainImages = m_Device->getSwapchainImagesKHR(m_SwapChain.get()).value;
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

        vk::RenderPassCreateInfo renderPassInfo{
            .sType = vk::StructureType::eRenderPassCreateInfo,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
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
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr,
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
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f,
        };

        vk::PipelineMultisampleStateCreateInfo multisamplingInfo{
            .sType = vk::StructureType::ePipelineMultisampleStateCreateInfo,
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
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
            .blendConstants = vk::ArrayWrapper1D<float, 4Ui64>({ 0.0f, 0.0f, 0.0f, 0.0f }),
        };

        vk::DynamicState dynamicStates[]{
            vk::DynamicState::eViewport,
            vk::DynamicState::eLineWidth
        };

        vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
            .sType = vk::StructureType::ePipelineDynamicStateCreateInfo,
            .dynamicStateCount = 2,
            .pDynamicStates = dynamicStates,
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = vk::StructureType::ePipelineLayoutCreateInfo,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        };

        auto [result, pipelineLayout] = m_Device->createPipelineLayoutUnique(pipelineLayoutInfo);

        m_PipelineLayout = std::move(pipelineLayout);

        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        m_Device->destroyShaderModule(vertShaderModule);
        m_Device->destroyShaderModule(fragShaderModule);
    }

    void MainLoop() {
        while (!glfwWindowShouldClose(m_Window)) {
            glfwPollEvents();
        }
    }

    void Cleanup() {
        m_Device.release();
        m_Surface.release();
        m_DebugMessenger.release();
        m_Instance.release();

        glfwDestroyWindow(m_Window);

        glfwTerminate();
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