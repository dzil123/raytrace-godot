shader_type canvas_item;
const float PI = 3.1415926535897932384626433;
const float ES = 0.0000001; // epsilon

// http://three-eyed-games.com/2018/05/03/gpu-ray-tracing-in-unity-part-1/

// camera coordinates to homogenous coords
// https://github.com/g-truc/glm/blob/416fa93e42f8fe1d85a93888a113fecd79e01453/glm/ext/matrix_clip_space.inl#L239
mat4 perspective(float fovy, float aspect, float zNear, float zFar) {
    float tanHalfFovy = tan(radians(fovy) / 2.0);
    
    float A = 1.0 / (aspect * tanHalfFovy);
    float B = 1.0 / (tanHalfFovy);
    float C = -1.0 * (zFar + zNear) / (zFar - zNear);
    float D = (-2.0 * zFar * zNear) / (zFar - zNear);
	
	mat4 matrix = mat4(
		vec4(A,  0., 0., 0.),
		vec4(0., B,  0., 0.),
		vec4(0., 0., C, -1.),
		vec4(0., 0., D,  0.)
	);
	
    return matrix;
}

// TODO: try alternative to lookAt
// world to camera
// https://github.com/g-truc/glm/blob/0b974f0d00a0e45bd376733be96488d0986e05eb/glm/ext/matrix_transform.inl#L99
mat4 lookAt(vec3 eye, vec3 center, vec3 up) { // position of camera, coordinate to look at, up direction
	vec3 f = normalize(center - eye); // forward
	vec3 s = normalize(cross(f, up)); // side, aka right
	vec3 u = cross(s, f);             // up
	
	mat4 matrix = mat4(
		vec4(s, 0.),
		vec4(u, 0.),
		vec4(-f, 0.),
		vec4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.)
	);
	
	return matrix;
}

float saturate(float f) {
	return clamp(f, 0.0, 1.0);
}

struct Ray {
	vec3 pos; // position
	vec3 dir; // direction (normalized)
};

struct RayHit {
	vec3 pos; // position
	float dist; // distance
	vec3 normal;
};

struct Sphere {
	vec3 pos;
	float radius;
//	vec3 color;
};

const int NUM_SPHERES = 2;
struct Scene {
	float ground;
	Sphere spheres[2]; // cannot put NUM_SPHERES here
};

Scene getScene() {
	Scene scene;
	scene.ground = 0.;
	
	for (int i = 0; i < NUM_SPHERES; i++) {
		scene.spheres[i] = Sphere(vec3(0.), -1.0);
	}
	scene.spheres[0] = Sphere(vec3(3.0, 1.5, 0.0), 0.5);
//	scene.spheres[0] = Sphere(vec3(0.0), 0.5);
	
	return scene;
}

vec3 cameraOrigin() {
	vec3 origin = vec3(0., 1., 0.);
//	vec3 origin = vec3(5.0 * sin(TIME), 2.0, 0.0);
//	origin = vec3(0.0, 1.0, -1.5);
//	origin.x += -cos(TIME*1.+.7) + 0.;

//	float tscale = 1.0;
//	origin.xz = vec2(sin(TIME/tscale), cos(TIME/tscale)) * 2.0;
	
	return origin;
}

vec3 cameraLook() {
	vec3 pos = vec3(10.0, 3.0, 4.0);
	
	float scale = 1.0;
	float tscale = 5.;
	
	pos.x = sin(TIME / tscale) * scale;
	pos.y = 1.5 + sin(TIME * 16. / tscale) * 0.25;
	pos.z = -cos(TIME / tscale) * scale;

//	return vec3(0., 1., 0.);
	return vec3(3.0, 1.5, 0.0);
//	return vec3(3.0, 1.5, 0.0) + pos / 2.;
	
	return pos;
}

Ray cameraRay(vec2 uv) {
	vec3 origin = cameraOrigin();
	vec3 pos = cameraLook();
	mat4 camToWorld = lookAt(origin, pos, vec3(0., -1., 0.));
//	origin = (camToWorld * vec4(vec3(0.), 1.)).xyz; // recompute origin from camToWorld, but not needed
	
	mat4 matrix = perspective(90, 16.0 / 9.0, .1, 10.0);
	matrix = inverse(matrix); // homogenous to camera
	
	vec3 direction = (matrix * vec4(uv, 0., 1.)).xyz; // convert screen coord (depth = 0, w=1 bc position) into camera direction
	direction = (camToWorld * vec4(direction, 0.)).xyz; // convert camera direction into world direction
	direction = normalize(direction);
	
	return Ray(origin, direction);
}

RayHit newRayHit() {
	return RayHit(vec3(0.), -1.0, vec3(0., 1., 0.));
}

float groundDistance(Ray ray, float ground) {
	return -(ray.pos.y - ground) / ray.dir.y; // negative = no hit
}

void groundIntersect(Ray ray, float ground, inout RayHit hit) {
	float dist = groundDistance(ray, ground);
	
	if (dist > 5.) {
		return;
	}
	
	if ((dist > 0.0) && (dist < hit.dist || hit.dist < 0.)) {
		hit.dist = dist;
		hit.pos = ray.pos + (dist * ray.dir);
		hit.normal = vec3(0., 1., 0.);
	}
}

void sphereIntersect(Ray ray, Sphere sphere, inout RayHit hit) {
	vec3 d = ray.pos - sphere.pos;
	float p1 = -dot(ray.dir, d);
	float p2sqr = p1 * p1 - dot(d, d) + sphere.radius * sphere.radius;
	if (p2sqr < 0.) {
		return;
	}
	float p2 = sqrt(p2sqr);
	
	float dist = (p1 - p2 > 0.) ? (p1 - p2) : (p1 + p2);
	if ((dist > 0.0) && (dist < hit.dist || hit.dist < 0.)) {
		hit.dist = dist;
		hit.pos = ray.pos + (dist * ray.dir);
		hit.normal = normalize(hit.pos - sphere.pos);
	}
}

RayHit trace(Ray ray) {
	Scene scene = getScene();
	RayHit hit = newRayHit();
		
	groundIntersect(ray, scene.ground, hit);
	
	for(int i = 0; i < NUM_SPHERES; i++) {
		Sphere sphere = scene.spheres[i];
		if (sphere.radius < 0.) {
			break;
		}
		sphereIntersect(ray, sphere, hit);
	}
	
	return hit;
}

// convert ray direction to equirectangular mapping
vec2 hdri_pos(Ray ray) {
	float theta = acos(ray.dir.y) / PI;
	float phi = atan(ray.dir.x, -ray.dir.z) / PI + .5;
	return vec2(phi, theta); // ensure -1 <= x <= 1
}

vec3 hdri_lookup(sampler2D tex, Ray ray) {
	return texture(tex, hdri_pos(ray)).xyz;
}

vec3 shadeHdri(vec3 color) {
//	return color;
//	return pow(color, vec3(1. / 2.2));
	
	// clamp highs
	vec3 bias = vec3(1.0);
	color = min(color, bias) / bias;
	
	// lighten image, compress dynamic range
	const int iterations = 6;
	for (int i = 0; i < iterations; i++) {
		color = log2(color + 1.0);
	}
	
	return color;
}

vec3 shade(Ray ray, RayHit hit, vec3 hdri) {
	vec3 hdriShaded = shadeHdri(hdri);
	vec3 color = vec3(0.);
	
	/*
		float ground = groundDistance(ray, 0.);
		vec3 nice = vec3(0.6, 0.8, 0.8);
		if (ground > 0.) {
			float f = ground;
			f = log(f) \ log(4.);
			f = saturate(f);
			f = 1. - f;
			color = mix(nice, color, f);
		} else {
			color = nice * .95;
	*/
	
	if (hit.dist > 0.) {
		color = hit.normal * 0.5 + 0.5;
		
		float f = hit.dist;
		f = log(f) / log(10.);
		f = saturate(f);
		f = 1. - f;
//		color *= f;
		
//		color = hit.pos*.5 + .5;
		
		color = mix(hdriShaded, color, 0.5);
	} else {
		color = hdriShaded;
	}
	
	return color;
}

void fragment() {
	vec2 uv = UV;
//	uv.y = 1. - uv.y;
	uv = uv * 2. - 1.;
	
	vec3 color = vec3(0.);
	
	const int iters = 1;
	const vec2 offsets[5] = {vec2(0.), vec2(-0.5, 0.), vec2(0.5, 0.), vec2(0., -0.5), vec2(.0, 0.5)};
	
	for(int i = 0; i < iters; i++) {
//		Ray ray = cameraRay(uv + SCREEN_PIXEL_SIZE.xy * offsets[i]);
		Ray ray = cameraRay(uv);
		RayHit hit = trace(ray);

	//	vec3 hdri = texture(TEXTURE, hdri(ray)).xyz; // TEXTURE only works in fragment()
		vec3 hdri = hdri_lookup(TEXTURE, ray); // TEXTURE only works in fragment()
		vec3 _color = shade(ray, hit, hdri);

		color += _color / float(iters);
	}
	
//	color = vec3(uv, 0.);
	COLOR = vec4(color, 1.0);
}
