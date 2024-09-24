workspace "Vulkan-Basic"
	language "C++"
	cppdialect "C++17"
	--cppdialect "C++20"

	platforms { "x64" }
	configurations { "debug", "release" }

	flags "NoPCH"
	flags "MultiProcessorCompile"

	startproject "src"

	debugdir "%{wks.location}"
	objdir "_build_/%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	targetsuffix "-%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	
	-- Default toolset options
	filter "toolset:gcc or toolset:clang"
		linkoptions { "-pthread" }
		buildoptions { "-march=native", "-Wall", "-pthread" }

	filter "toolset:msc-*"
		defines { "_CRT_SECURE_NO_WARNINGS=1" }
		defines { "_SCL_SECURE_NO_WARNINGS=1" }
		buildoptions { "/utf-8" }
	
	filter "*"

	-- default libraries
	filter "system:linux"
		links "dl"
	
	filter "system:windows"

	filter "*"

	-- default outputs
	filter "kind:StaticLib"
		targetdir "lib/"

	filter "kind:ConsoleApp"
		targetdir "bin/"
		targetextension ".exe"
	
	filter "*"

	--configurations
	filter "debug"
		symbols "On"
		defines { "_DEBUG=1" }

	filter "release"
		optimize "On"
		defines { "NDEBUG=1" }

	filter "*"

-- Third party dependencies
include "third_party" 

-- GLSLC helpers
dofile( "util/glslc.lua" )

-- Projects
project "basic-vulkan"
	local sources = { 
		"src/**.cpp",
		"src/**.hpp",
		"src/**.hxx"
	}

	kind "ConsoleApp"
	location "src"

	files( sources )

	dependson "shaders"

	links "labutils"
	links "x-volk"
	links "x-stb"
	links "x-glfw"
	links "x-vma"
	links "imgui"

	dependson "x-glm" 
	dependson "x-rapidobj"

project "shaders"
	local shaders = { 
		"src/shaders/*.vert",
		"src/shaders/*.frag",
		"src/shaders/*.comp",
		"src/shaders/*.geom",
		"src/shaders/*.tesc",
		"src/shaders/*.tese"
	}

	kind "Utility"
	location "src/shaders"

	files( shaders )

	handle_glsl_files( "-O", "assets/src/shaders", {} )

project "labutils"
	local sources = { 
		"labutils/**.cpp",
		"labutils/**.hpp",
		"labutils/**.hxx"
	}

	kind "StaticLib"
	location "labutils"

	files( sources )

--EOF
