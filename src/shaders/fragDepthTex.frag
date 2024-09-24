#version 450

layout (location = 0) in vec2 v2fTexCoord;

layout(set = 1, binding = 0) uniform sampler2D uTexColor;

layout(location = 0) out vec4 oColor;

void main()
{
	//Get fragment depth from gl_FragCoord.z
	//All values are close together, so spread them out
	float linearDepthValue = (gl_FragCoord.z - 0.90) / 0.10;
	
	//Ensure the value is between 0 and 1
	float t = clamp(linearDepthValue, 0.0, 1.0);

	//Set colour to be a gradient of black and white
	vec3 color = mix(vec3(1.0), vec3(0.0), t);

	oColor = vec4(color, 1.0);

}