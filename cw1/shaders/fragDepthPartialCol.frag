#version 450

layout (location = 0) in vec3 color;


layout (location = 0) out vec4 oColor;

void main()
{
	//Partial Derivatives of Fragment Depth
	float dx = dFdx(gl_FragCoord.z);
	float dy = dFdy(gl_FragCoord.z);

	//Normalise the derivatives
	vec2 derivatives = vec2(dx, dy);
	vec2 normalisedDerivatives = normalize(derivatives);

	//Ensure that the values are between 0 and 1 (so we can render colours)
	normalisedDerivatives = (normalisedDerivatives + 1) / 2;


	oColor = vec4(normalisedDerivatives.x, normalisedDerivatives.y, 0.0, 1.0);

}