#include <volk/volk.h>

#include <tuple>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "load_model_obj.hpp"
#include "simple_model.hpp"




namespace
{
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
		// See sources in exercise4/shaders/*. 
#		define SHADERDIR_ "assets/cw1/shaders/"
		constexpr char const* kTextureVertShaderPath = SHADERDIR_ "defaultTex.vert.spv";
		constexpr char const* kTextureFragShaderPath = SHADERDIR_ "defaultTex.frag.spv";

		constexpr char const* kColourVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kColourFragShaderPath = SHADERDIR_ "default.frag.spv";

		//Rendering modes
		constexpr char const* kTexFragMipmapShaderPath = SHADERDIR_ "fragMipmapTex.frag.spv";

		constexpr char const* kTexFragDepthShaderPath = SHADERDIR_ "fragDepthTex.frag.spv";
		constexpr char const* kColFragDepthShaderPath = SHADERDIR_ "fragDepthCol.frag.spv";

		constexpr char const* kTexFragDepthPartialShaderPath = SHADERDIR_ "fragDepthPartialTex.frag.spv";
		constexpr char const* kColFragDepthPartialShaderPath = SHADERDIR_ "fragDepthPartialCol.frag.spv";

#		undef SHADERDIR_

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;

		//More camera settings, useful for debug
		constexpr float kCameraBaseSpeed = 0.01f; //Units/second
		constexpr float kCameraFastMult = 2.f; //Speed multiplier
		constexpr float kCameraSlowMult = 0.05f; //Speed multiplier

		constexpr float kCameraMouseSensitivity = 0.01f; //Radians per pixel
	}

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	// Uniform data
	namespace glsl
	{
		struct SceneUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");
	}

	// Helpers:
	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		max
	};

	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};

	//Helpful structs
	struct ColorizedMesh
	{
		labutils::Buffer positions;
		labutils::Buffer colors;

		std::uint32_t vertexCount;
	};

	struct ColouredMeshDetails
	{
		std::vector<float> positions;
		std::vector<float> colours;
		size_t vertexCount = 0;
	};

	struct TexturedMesh
	{
		labutils::Buffer positions;
		labutils::Buffer texcoords;

		std::uint32_t vertexCount;
	};

	struct TexturedMeshDetails
	{
		std::vector<float> positions;
		std::vector<float> texCoords;
		size_t vertexCount = 0;
	};



	void update_user_state(UserState&, float aElapsedTime);

	lut::RenderPass create_render_pass(lut::VulkanWindow const&);
	//lut::RenderPass create_storage_render_pass(lut::VulkanWindow const& aWindow);

	TexturedMesh create_textured_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, float aPositions[], float aTexCoords[], size_t aVertCount);
	ColorizedMesh create_coloured_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, float aPositions[], float aColor[], size_t aVertCount);

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const&);
	//lut::DescriptorSetLayout create_storage_descriptor_layout(lut::VulkanWindow const&);

	lut::PipelineLayout create_coloured_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout);
	lut::PipelineLayout create_textured_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout);
	//lut::PipelineLayout create_storage_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout);

	lut::Pipeline create_coloured_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout, const char* vertShaderPath, const char* fragShaderPath);
	lut::Pipeline create_textured_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout, const char* vertShaderPath, const char* fragShaderPath);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const&
	);

	void submit_commands(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
	void present_results(
		VkQueue,
		VkSwapchainKHR,
		std::uint32_t aImageIndex,
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);
}

int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	// Configure the GLFW window
	UserState state{};

	glfwSetWindowUserPointer(window.window, &state);
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass(window);

	//Create scene descriptor set layout
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);

	//Create object descriptor set layout
	lut::DescriptorSetLayout objectLayout = create_object_descriptor_layout(window);

	lut::PipelineLayout texturedPipeLayout = create_textured_pipeline_layout(window, sceneLayout.handle, objectLayout.handle);
	lut::PipelineLayout colouredPipeLayout = create_coloured_pipeline_layout(window, sceneLayout.handle);

	lut::Pipeline colouredPipe, texturedPipe;

#if defined(MAIN) || defined(ANISOTROPIC)
	colouredPipe = create_coloured_pipeline(window, renderPass.handle, colouredPipeLayout.handle, cfg::kColourVertShaderPath, cfg::kColourFragShaderPath);
	texturedPipe = create_textured_pipeline(window, renderPass.handle, texturedPipeLayout.handle, cfg::kTextureVertShaderPath, cfg::kTextureFragShaderPath);
#endif // MAIN

#ifdef MIPMAP
	colouredPipe = create_coloured_pipeline(window, renderPass.handle, colouredPipeLayout.handle, cfg::kColourVertShaderPath, cfg::kColourFragShaderPath);
	texturedPipe = create_textured_pipeline(window, renderPass.handle, texturedPipeLayout.handle, cfg::kTextureVertShaderPath, cfg::kTexFragMipmapShaderPath);
#endif

#ifdef FRAGDEPTH
	colouredPipe = create_coloured_pipeline(window, renderPass.handle, colouredPipeLayout.handle, cfg::kColourVertShaderPath, cfg::kColFragDepthShaderPath);
	texturedPipe = create_textured_pipeline(window, renderPass.handle, texturedPipeLayout.handle, cfg::kTextureVertShaderPath, cfg::kTexFragDepthShaderPath);
#endif

#ifdef FRAGDEPTHPARTIAL
	colouredPipe = create_coloured_pipeline(window, renderPass.handle, colouredPipeLayout.handle, cfg::kColourVertShaderPath, cfg::kColFragDepthPartialShaderPath);
	texturedPipe = create_textured_pipeline(window, renderPass.handle, texturedPipeLayout.handle, cfg::kTextureVertShaderPath, cfg::kTexFragDepthPartialShaderPath);
#endif

	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);


	//Load the mesh
	SimpleModel meshes = load_simple_wavefront_obj("assets/cw1/sponza_with_ship.obj");

	//Data structure to store all ColourizedMeshes
	std::vector<ColorizedMesh> colouredMeshes;

	//Data structure to store all TexturedMeshes
	std::vector<const char*> meshMaterial;
	std::vector<TexturedMesh> texturedMeshes;

	for (SimpleMeshInfo mesh : meshes.meshes)
	{

		size_t meshSize = mesh.vertexCount;
		size_t start = mesh.vertexStartIndex;

		//Only perform on textured meshes
		if (mesh.textured)
		{
			TexturedMeshDetails meshDetails;

			for (size_t i = start; i < start + meshSize; i++)
			{
				meshDetails.positions.emplace_back(meshes.dataTextured.positions.at(i).x);
				meshDetails.positions.emplace_back(meshes.dataTextured.positions.at(i).y);
				meshDetails.positions.emplace_back(meshes.dataTextured.positions.at(i).z);

				meshDetails.texCoords.emplace_back(meshes.dataTextured.texcoords.at(i).x);
				meshDetails.texCoords.emplace_back(meshes.dataTextured.texcoords.at(i).y);

			}

			meshDetails.vertexCount = meshSize;

			meshMaterial.emplace_back(meshes.materials[mesh.materialIndex].diffuseTexturePath.c_str());

			texturedMeshes.emplace_back(create_textured_mesh(window, allocator, meshDetails.positions.data(), meshDetails.texCoords.data(), meshDetails.vertexCount));

		}

		//Otherwise the mesh is coloured
		else
		{
			ColouredMeshDetails meshDetails;

			for (size_t i = start; i < start + meshSize; i++)
			{
				meshDetails.positions.emplace_back(meshes.dataUntextured.positions.at(i).x);
				meshDetails.positions.emplace_back(meshes.dataUntextured.positions.at(i).y);
				meshDetails.positions.emplace_back(meshes.dataUntextured.positions.at(i).z);

				meshDetails.colours.emplace_back(meshes.materials[mesh.materialIndex].diffuseColor.r);
				meshDetails.colours.emplace_back(meshes.materials[mesh.materialIndex].diffuseColor.g);
				meshDetails.colours.emplace_back(meshes.materials[mesh.materialIndex].diffuseColor.b);

			}

			meshDetails.vertexCount = meshSize;

			colouredMeshes.emplace_back(create_coloured_mesh(window, allocator, meshDetails.positions.data(), meshDetails.colours.data(), meshDetails.vertexCount));

		}

	}


	//Create SceneUniform Buffer
	lut::Buffer sceneUBO = lut::create_buffer(allocator, sizeof(glsl::SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

	//Create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	//Allocate descriptor set for uniform buffer
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	//Load textures into image
	lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

	//Create image view for texture image
	std::vector<lut::Image> images(texturedMeshes.size());

	//Create images
	for (size_t i = 0; i < texturedMeshes.size(); i++)
	{
		const char* path = meshMaterial[i];
		images[i] = lut::load_image_texture2d(path, window, cpool.handle, allocator);
	}

	std::vector<lut::ImageView> imageViews(images.size());

	//Create image views
	for (size_t i = 0; i < images.size(); i++)
	{
		imageViews[i] = lut::create_image_view_texture2d(window, images.at(i).image, VK_FORMAT_R8G8B8A8_SRGB);
	}

	//Create default texture sampler (required for descriptor set)
	lut::Sampler defaultSampler = lut::create_default_sampler(window);

	//Create descriptor sets
	std::vector<VkDescriptorSet> meshDescriptorSets(texturedMeshes.size());

	for (size_t i = 0; i < meshDescriptorSets.size(); i++)
	{
		meshDescriptorSets[i] = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);

		//Update descriptor set
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorImageInfo textureInfo{};
		textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo.imageView = imageViews.at(i).handle;
		textureInfo.sampler = defaultSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = meshDescriptorSets[i];
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &textureInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);

		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

#ifdef OVERDRAW
	//Setting up storage image for overdrawing visualization
	lut::Image storageImage;

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_R32_UINT; //Only need one channel (number of shader occurences)
	imageInfo.extent.width = window.swapchainExtent.width;
	imageInfo.extent.height = window.swapchainExtent.height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT; //Used for storage
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	//Create the image
	if (auto const res = vkCreateImage(window.device, &imageInfo, nullptr, &storageImage.image); VK_SUCCESS != res)
	{
		throw lut::Error("Unable to create image\n" "vkCreateImage() returned %s", lut::to_string(res).c_str());
	}

	//Allocate memory for the newly created image
	VmaAllocationCreateInfo allocInfo{};
	allocInfo.flags = 0;
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;


	if (auto const res = vmaCreateImage(allocator.allocator, &imageInfo, &allocInfo, &storageImage.image, &storageImage.allocation, nullptr); VK_SUCCESS != res)
	{
		throw lut::Error("Unable to allocate storage image\n" "vmaCreateImage returned %s", lut::to_string(res).c_str());
	}

	//Create image view
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = storageImage.image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_R32_UINT;
	viewInfo.components = VkComponentMapping{}; // == Identity
	viewInfo.subresourceRange = VkImageSubresourceRange{
		VK_IMAGE_ASPECT_COLOR_BIT,
		0, 1,
		0, 1
	};

	lut::ImageView view;

	if (auto const res = vkCreateImageView(window.device, &viewInfo, nullptr, &view.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Unable to allocate storage image view\n" "vkCreateImageView returned %s", lut::to_string(res).c_str());
	}

	//Create a layout for the descriptor set
	lut::DescriptorSetLayout storageLayout = create_storage_descriptor_layout(window);

	//Create render pass
	lut::RenderPass storagePass = create_storage_render_pass(window);

	//Create storage pipeline layout
	lut::PipelineLayout storagePipeLayout = create_storage_pipeline_layout(window, storageLayout.handle);

	//Create a pipeline using the layout
	lut::Pipeline storagePipe = create_storage_pipeline(window, storagePass.handle, storagePipeLayout.handle, fds, sdf);

	//Create storage descriptor set
	//Allocate descriptor set for uniform buffer
	VkDescriptorSet storageDescriptorSet = lut::alloc_desc_set(window, dpool.handle, storageLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorImageInfo storageInfo{};
		storageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		storageInfo.imageView = view.handle;
		storageInfo.sampler = VK_NULL_HANDLE; //No need for a sampler

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = storageDescriptorSet;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &storageInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

#endif


	// Application main loop
	bool recreateSwapchain = false;

	//Record time before main loop starts
	auto previousClock = Clock_::now();

	while (!glfwWindowShouldClose(window.window))
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if (recreateSwapchain)
		{
			//Wait for the GPU to finish processing
			vkDeviceWaitIdle(window.device);

			//Recreate them
			auto const changes = lut::recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);

			if (changes.changedSize)
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			if (changes.changedSize)
			{
#if defined(MAIN) || defined(ANISOTROPIC)
				colouredPipe = create_coloured_pipeline(window, renderPass.handle, colouredPipeLayout.handle, cfg::kColourVertShaderPath, cfg::kColourFragShaderPath);
				texturedPipe = create_textured_pipeline(window, renderPass.handle, texturedPipeLayout.handle, cfg::kTextureVertShaderPath, cfg::kTextureFragShaderPath);
#endif 

#ifdef MIPMAP
				colouredPipe = create_coloured_pipeline(window, renderPass.handle, colouredPipeLayout.handle, cfg::kColourVertShaderPath, cfg::kColourFragShaderPath);
				texturedPipe = create_textured_pipeline(window, renderPass.handle, texturedPipeLayout.handle, cfg::kTextureVertShaderPath, cfg::kTexFragMipmapShaderPath);
#endif


#ifdef FRAGDEPTH
				colouredPipe = create_coloured_pipeline(window, renderPass.handle, colouredPipeLayout.handle, cfg::kColourVertShaderPath, cfg::kColourFragShaderPath);
				texturedPipe = create_textured_pipeline(window, renderPass.handle, texturedPipeLayout.handle, cfg::kTextureVertShaderPath, cfg::kTexFragDepthShaderPath);
#endif

#ifdef FRAGDEPTHPARTIAL
				colouredPipe = create_coloured_pipeline(window, renderPass.handle, colouredPipeLayout.handle, cfg::kColourVertShaderPath, cfg::kColourFragShaderPath);
				texturedPipe = create_textured_pipeline(window, renderPass.handle, texturedPipeLayout.handle, cfg::kTextureVertShaderPath, cfg::kTexFragDepthPartialShaderPath);
#endif

			}

			recreateSwapchain = false;
			continue;
		}

		//Acquire next swapchain image
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(window.device, window.swapchain, std::numeric_limits<std::uint64_t>::max(), imageAvailable.handle, VK_NULL_HANDLE, &imageIndex);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire next swapchain image" "vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}

		//Wait for command buffer to be available
		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n" "vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n" "vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		//Record and submit commands
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		//Update state
		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();

		update_user_state(state, dt);

		//Prepare data for this frame
		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);

		//Record commands
		//Begin recording commands

		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(cbuffers[imageIndex], &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Update unifom buffer
		lut::buffer_barrier(cbuffers[imageIndex], sceneUBO.buffer, VK_ACCESS_UNIFORM_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(cbuffers[imageIndex], sceneUBO.buffer, 0, sizeof(glsl::SceneUniform), &sceneUniforms);

		lut::buffer_barrier(cbuffers[imageIndex], sceneUBO.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

		//Begin render pass
		//Clear to a dark gray background
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.f;

		clearValues[1].depthStencil.depth = 1.f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = renderPass.handle;
		passInfo.framebuffer = framebuffers[imageIndex].handle;
		passInfo.renderArea.offset = VkOffset2D{ 0,0 };
		passInfo.renderArea.extent = VkExtent2D{ window.swapchainExtent.width, window.swapchainExtent.height };
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(cbuffers[imageIndex], &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		//Draw with the textured pipeline
		vkCmdBindPipeline(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, texturedPipe.handle);

		//Bind the descriptors
		vkCmdBindDescriptorSets(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, texturedPipeLayout.handle, 0, 1, &sceneDescriptors, 0, nullptr);

		//Bind the textured meshes and draw
		for (size_t i = 0; i < texturedMeshes.size(); i++)
		{
			vkCmdBindDescriptorSets(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, texturedPipeLayout.handle, 1, 1, &meshDescriptorSets[i], 0, nullptr);

			VkBuffer meshBuffers[2] = { texturedMeshes.at(i).positions.buffer, texturedMeshes.at(i).texcoords.buffer };
			VkDeviceSize meshOffsets[2] = {};

			vkCmdBindVertexBuffers(cbuffers[imageIndex], 0, 2, meshBuffers, meshOffsets);

			vkCmdDraw(cbuffers[imageIndex], texturedMeshes.at(i).vertexCount, 1, 0, 0);

		}

		//Now bind the coloured meshes
		vkCmdBindPipeline(cbuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, colouredPipe.handle);

		//Draw all coloured meshes
		for (size_t i = 0; i < colouredMeshes.size(); i++)
		{
			VkBuffer meshBuffers[2] = { colouredMeshes.at(i).positions.buffer, colouredMeshes.at(i).colors.buffer };
			VkDeviceSize meshOffsets[2] = {};

			vkCmdBindVertexBuffers(cbuffers[imageIndex], 0, 2, meshBuffers, meshOffsets);

			vkCmdDraw(cbuffers[imageIndex], colouredMeshes.at(i).vertexCount, 1, 0, 0);
		}

		//End the render pass
		vkCmdEndRenderPass(cbuffers[imageIndex]);

		//End command recording
		if (auto const res = vkEndCommandBuffer(cbuffers[imageIndex]); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Submit the recorded commands
		submit_commands(window, cbuffers[imageIndex], cbfences[imageIndex].handle, imageAvailable.handle, renderFinished.handle);

		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);

	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.

	vkDeviceWaitIdle(window.device);

	return 0;
}
catch (std::exception const& eErr)
{
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, "Error: %s\n", eErr.what());
	return 1;
}

namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}

		//Handle camera controls
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;

		default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			//Only update rotation on second frame of mouse navigation
			//Ensures previous X and Y variables are initialised to sensible values
			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));

			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;

			aState.wasMousing = true;
		}

		else
		{
			aState.wasMousing = false;
		}

		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));
	}
}

namespace
{
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& aState)
	{
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);

		aSceneUniforms.projection[1][1] *= -1.f; //Mirror the y axis

		aSceneUniforms.camera = glm::inverse(aState.camera2world);
		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
	}
}

namespace
{
	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		//Declare the single subpass
		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1; //refers to attachments[1]
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		//Introduce a dependency
		VkSubpassDependency deps[2]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcAccessMask = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[1].dstSubpass = 0;
		deps[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;


		//With declarations in place, we can now create the render pass
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2;
		passInfo.pDependencies = deps;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}
	/*
	lut::RenderPass create_storage_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[1]{};
		attachments[0].format = VK_FORMAT_R32_UINT;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE; //Set to compute for counting

		//With declarations in place, we can now create the render pass
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 1;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}
	*/

	lut::PipelineLayout create_coloured_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout)
	{
		VkDescriptorSetLayout layouts[] =
		{
			aSceneLayout,  //Order must match the set = N in the shaders - aSceneLayout = set 0
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create coloured pipeline layout\n" "vkCreatePipelineLayout returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_textured_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout)
	{
		VkDescriptorSetLayout layouts[] =
		{
			aSceneLayout,
			aObjectLayout //Order must match the set = N in the shaders - aSceneLayout = set 0
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create textured pipeline layout\n" "vkCreatePipelineLayout returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	/*lut::PipelineLayout create_storage_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aStorageLayout)
	{
		VkDescriptorSetLayout layouts[] =
		{
			aStorageLayout
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create storage pipeline layout\n" "vkCreatePipelineLayout returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}*/


	lut::Pipeline create_coloured_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, const char* vertShaderPath, const char* fragShaderPath)
	{
		//Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, vertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, fragShaderPath);

		//Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		//Define vertex input attributes
		VkVertexInputBindingDescription vertexInputs[2]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 3;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[2]{};
		vertexAttributes[0].binding = 0; //Must match binding above
		vertexAttributes[0].location = 0; //Must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; //Must match binding above
		vertexAttributes[1].location = 1; //Must match shader
		vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[1].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 2; //Number of vertexInputs
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 2; //Number of vertexAttributes
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		//Define which primitive the input is assembled into for rasterization
		//in this case, it's triangles
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Define rasterisation options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Define multi sampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//Define blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//Define depth stencil state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		//We can now create the pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create coloured pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_textured_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, const char* vertShaderPath, const char* fragShaderPath)
	{
		//Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, vertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, fragShaderPath);

		//Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		//Define vertex input attributes
		VkVertexInputBindingDescription vertexInputs[2]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[2]{};
		vertexAttributes[0].binding = 0; //Must match binding above
		vertexAttributes[0].location = 0; //Must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; //Must match binding above
		vertexAttributes[1].location = 1; //Must match shader
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 2; //Number of vertexInputs
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 2; //Number of vertexAttributes
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		//Define which primitive the input is assembled into for rasterization
		//in this case, it's triangles
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Define rasterisation options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f;

		//Define multi sampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//Define blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_TRUE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//Define depth stencil state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		//We can now create the pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create textured pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}
	/*
	lut::Pipeline create_storage_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, const char* vertShaderPath, const char* fragShaderPath)
	{
		//Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, vertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, fragShaderPath);

		//Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		//We can now create the pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create textured pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}
	*/
	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] =
			{
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n" "vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());
			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; //Number must match the index of the corresponding *binding = N* declaration in shader
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);

	}
	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; //Number must match the index of the corresponding *binding = N* declaration in shader
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	/*lut::DescriptorSetLayout create_storage_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; //Number must match the index of the corresponding *binding = N* declaration in shader
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}
	*/

	void submit_commands(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	void present_results(VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain)
	{
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);
		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}

		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable to present swapchain image %u\n" "vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}
	}
}

namespace
{
	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n" "vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		//Create image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1,
			0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n" "vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}
}

namespace
{
	TexturedMesh create_textured_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, float aPositions[], float aTexCoords[], size_t aVertCount)
	{

		VkDeviceSize posSize = aVertCount * 3 * sizeof(float);
		VkDeviceSize texSize = aVertCount * 2 * sizeof(float);

		//Create final position and colour buffers
		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			posSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0, //No additional VmaAllocationCreateFlags
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE //Can also be VMA_MEMORY_USAGE_AUTO
		);

		lut::Buffer vertexTexGPU = lut::create_buffer(
			aAllocator,
			texSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0, //No additional VmaAllocationCreateFlags
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE //Can also be VMA_MEMORY_USAGE_AUTO
		);

		//Create staging buffers
		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			posSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer texStaging = lut::create_buffer(
			aAllocator,
			texSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(posPtr, aPositions, posSize);
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		void* colPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, texStaging.allocation, &colPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(colPtr, aTexCoords, texSize);
		vmaUnmapMemory(aAllocator.allocator, texStaging.allocation);

		//Prepare for issuing the transfer commands that copy data from staging buffers to final on-GPU buffers
		//First, ensure that Vulkan resources are alive until all transfers are completed
		lut::Fence uploadComplete = lut::create_fence(aContext);

		//Queue data uploads from staging buffers to final buffers
		//Use a separate command pool for simplicity
		lut::CommandPool uploadPool = lut::create_command_pool(aContext);
		VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

		//Record copy commands into command buffer
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy pcopy{};
		pcopy.size = posSize;

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy tcopy{};
		tcopy.size = texSize;

		vkCmdCopyBuffer(uploadCmd, texStaging.buffer, vertexTexGPU.buffer, 1, &tcopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexTexGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Submit transfer commands
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}

		//Wait for commands to finish before destroying temporary resources needed for transfers
		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}

		TexturedMesh texMesh;
		texMesh.positions = std::move(vertexPosGPU);
		texMesh.texcoords = std::move(vertexTexGPU);
		texMesh.vertexCount = std::uint32_t(aVertCount);
		return texMesh;

	}

	ColorizedMesh create_coloured_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, float aPositions[], float aColor[], size_t aVertCount)
	{
		VkDeviceSize posSize = aVertCount * 3 * sizeof(float);
		VkDeviceSize colSize = aVertCount * 3 * sizeof(float);

		//Create final position and colour buffers
		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			posSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0, //No additional VmaAllocationCreateFlags
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE //Can also be VMA_MEMORY_USAGE_AUTO
		);

		lut::Buffer vertexColGPU = lut::create_buffer(
			aAllocator,
			colSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0, //No additional VmaAllocationCreateFlags
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE //Can also be VMA_MEMORY_USAGE_AUTO
		);

		//Create staging buffers
		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			posSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		lut::Buffer colStaging = lut::create_buffer(
			aAllocator,
			colSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		);

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(posPtr, aPositions, posSize);
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		void* colPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
		}

		std::memcpy(colPtr, aColor, colSize);
		vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

		//Prepare for issuing the transfer commands that copy data from staging buffers to final on-GPU buffers
		//First, ensure that Vulkan resources are alive until all transfers are completed
		lut::Fence uploadComplete = lut::create_fence(aContext);

		//Queue data uploads from staging buffers to final buffers
		//Use a separate command pool for simplicity
		lut::CommandPool uploadPool = lut::create_command_pool(aContext);
		VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

		//Record copy commands into command buffer
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkBufferCopy pcopy{};
		pcopy.size = posSize;

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ccopy{};
		ccopy.size = colSize;

		vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

		lut::buffer_barrier(
			uploadCmd,
			vertexColGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		//Submit transfer commands
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}

		//Wait for commands to finish before destroying temporary resources needed for transfers
		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}

		ColorizedMesh colorMesh;
		colorMesh.positions = std::move(vertexPosGPU);
		colorMesh.colors = std::move(vertexColGPU);
		colorMesh.vertexCount = std::uint32_t(aVertCount);
		return colorMesh;
	}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
