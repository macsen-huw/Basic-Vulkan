#version 450

layout (location = 0) in vec2 v2fTexCoord;

layout(set = 1, binding = 0) uniform sampler2D uTexColor;

layout(location = 0) out vec4 oColor;

void main()
{
	vec4 level0Colour = {1.0, 0.0, 0.0, 1.0}; //Red 
	vec4 level1Colour = {1.0, 0.5, 0.0, 1.0}; //Orange
	vec4 level2Colour = {1.0, 1.0, 0.0, 1.0}; //Yellow
	vec4 level3Colour = {0.5, 1.0, 0.0, 1.0}; //Light Green
	vec4 level4Colour = {0.0, 1.0, 0.0, 1.0}; //Green
	vec4 level5Colour = {0.0, 1.0, 0.5, 1.0}; //Turquoise
	vec4 level6Colour = {0.0, 1.0, 1.0, 1.0}; //Light Blue
	vec4 level7Colour = {0.0, 0.5, 1.0, 1.0}; //Blue
	vec4 level8Colour = {0.0, 0.0, 1.0, 1.0}; //Dark Blue
	vec4 level9Colour = {0.5, 0.0, 1.0, 1.0}; //Purple
	vec4 level10Colour ={1.0, 1.0, 1.0, 1.0}; //White
	

	//Extract mipmap level
	vec2 lodInfo = textureQueryLod(uTexColor, v2fTexCoord);
	int mipmapLevel = int(floor(lodInfo.y));

	//Wrap around the levels back to the start
	mipmapLevel = mipmapLevel % 10;

	if(mipmapLevel == 0)
		oColor = level0Colour;
	else if (mipmapLevel == 1)
		oColor = level1Colour;
	else if (mipmapLevel == 2)
		oColor = level2Colour;
	else if (mipmapLevel == 3)
		oColor = level3Colour;
	else if (mipmapLevel == 4)
		oColor = level4Colour;
	else if (mipmapLevel == 5)
		oColor = level5Colour;
	else if (mipmapLevel == 6)
		oColor = level6Colour;
	else if (mipmapLevel == 7)
		oColor = level7Colour;
	else if (mipmapLevel == 8)
		oColor = level8Colour;
	else if (mipmapLevel == 9)
		oColor = level9Colour;
	else if(mipmapLevel == 10)
		oColor = level10Colour;

}