#include <metal_stdlib>
using namespace metal;

#define far 80.0
#define distanceBias 0.6
#define epsilon 0.001


float2x2 rotato(float a) {
  return float2x2(cos(a), -sin(a), sin(a), cos(a));
}

float box( float3 p, float3 b) {
  float3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float map(float3 rp, float time) {
  float res = 0.0;

  float3 p = rp - float3(1.0, -0.25, 4.0);
  float3 b = float3(1.0, 1.0, 1.0);

  p.xy = p.xy * rotato(p.z * 0.3);
  p.y += sin(p.z + time + p.x * 1.0) * 0.2;
  p.x += cos(p.y - p.z * 2.0 + time) * 0.3;

  p = fmod(p, b) - 0.5 * b;

  res = box(p, float3(0.033, 0.033, 2.0));
  return res;
}


float3 getNormal(float3 p, float time) {
  float2 e = float2(0.0035, -0.0035);
  return normalize(e.xyy * map(p + e.xyy, time) +
                   e.yyx * map(p + e.yyx, time) +
                   e.yxy * map(p + e.yxy, time) +
                   e.xxx * map(p + e.xxx, time));
}

// life's so rad
float3 colorado(float3 p) {
  p.xy = p.xy * rotato(p.z * 0.64);
  float3 color = mix(float3(0.0, 0.3, 1.3), float3(1.2, 1.2, 0.0),  smoothstep(0.0, 1.0, p.x));
  return color;
}

float3 color(float3 ro, float3 rd, float3 norm, float3 lp, float t) {
  // lights
  float3 ld = lp - ro;
  float lightDistance = max(length(ld), 0.001);
  float lightAttenuation = 1.0 / (1.0 + lightDistance * 0.2 + lightDistance * lightDistance * 0.1);

  ld /= lightDistance;

  float diffuse = max(dot(norm, ld), 0.0);
  float specular = pow(max(dot(reflect(-ld, norm), -rd), 0.0), 12.0);

  // colors
  float3 threadColor = colorado(ro);
  threadColor = 0.5 + 0.5 * cos(t + lp.xyx + float3(0.0, 2.0, 4.0));

  // action
  float3 sceneColor = (threadColor * (diffuse + 0.15) + float3(1.0, 1.0, 1.0) * specular * 1.2) * lightAttenuation;
  return sceneColor;
}

float trace(float3 ro, float3 rd, float time) {
  float t = 0.0, d = 0.0;
  for (int i = 0; i < 100; i++) {
    d = map(ro + rd * t, time);

    if(abs(d) < epsilon || t > far) {
      break;
    }

    t += d * distanceBias;
  }
  return t;
}

float traceRef(float3 ro, float3 rd, float time){
  float t = 0.0, d = 0.0;
  for (int i = 0; i < 64; i++){
    d = map(ro + rd * t, time);
    if(abs(d) < 0.0025 || t > far) break;
    t += d;
  }

  return t;
}

kernel void howWeDoingMan(texture2d<float, access::write> o[[texture(0)]],
                          constant float &time [[buffer(0)]],
                          constant float2 *touchEvent [[buffer(1)]],
                          constant int &numberOfTouches [[buffer(2)]],
                          ushort2 gid [[thread_position_in_grid]]) {

  int width = o.get_width();
  int height = o.get_height();
  float2 res = float2(width, height);
  float2 p = float2(gid.xy);
  float2 uv = 2.0 * float2(p - 0.5 * res) / res.y;

  float3 ro = float3(0.0);
  float3 rd = normalize(float3(uv, 2.0));
  ro.z -= time * 0.7;

  float3 lightPosition = ro + float3(0.0, 1.0, 0.0);

  // set the scene
  float t = trace(ro, rd, time);

  ro += rd * t;

  float3 norm = getNormal(ro, time);

  float3 col = color(ro, rd, norm, lightPosition, t);

  float foam = t;

  rd = reflect(rd, norm);
  t = traceRef(ro +  rd * .01, rd, time);

  ro += rd * t;

  norm = getNormal(ro, time);

  col += color(ro, rd, norm, lightPosition, t) * 0.25;

  foam = smoothstep(0.0, 0.15, foam / 130.0);
  col = mix(col, float3(0), foam);
  col *= smoothstep(2.0, 0.29, length(uv));

  float4 color = float4(sqrt(clamp(col, 0.0, 1.0)), 1.0);
  o.write(color, gid);
}
