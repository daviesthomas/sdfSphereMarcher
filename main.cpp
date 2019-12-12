#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/signed_distance.h>
#include <igl/read_triangle_mesh.h>
#include <igl/png/writePNG.h>
#include <igl/parallel_for.h>

#include <model.h>
#include <cmath>
#include <ctime>

#include "miniball.hpp"

const int MAX_MARCHING_STEPS = 1000;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;

// for precomputing sdf crap
Eigen::MatrixXd V;
Eigen::MatrixXi T,F;
igl::AABB<Eigen::MatrixXd,3> tree;
Eigen::MatrixXd FN,VN,EN;
Eigen::MatrixXi E;
Eigen::VectorXi EMAP;

MLP mlp;
float r = 0.5;
int QUERY_COUNTER = 0;
enum Shaders { outline, gray, phong };

void sphere_normalization(Eigen::MatrixXd &V, float target_radius){
  typedef double mytype;
  int n = V.rows();
  int d = 3;

  mytype** ap = new mytype*[n];
  // lets fill ap with our points
  for (int i=0; i<n; ++i) {
    mytype* p = new mytype[d];
    for (int j=0; j<d; ++j) {
      p[j] = V.coeff(i,j);  
    }
    ap[i]=p;
  }

  typedef mytype* const* PointIterator; 
  typedef const mytype* CoordIterator;
  typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MB;
  MB mb (3, ap, ap+n);

  // radius can be used to scale the shape!
  float radius = sqrt(mb.squared_radius()); 
  float scale = target_radius/radius;

  Eigen::Affine3d T = Eigen::Affine3d::Identity();
  T.translation() <<
    -mb.center()[0]*scale, 
    -mb.center()[1]*scale, 
    -mb.center()[2]*scale;

  T.scale(scale);

  for( auto i=0; i < V.rows(); ++i){
      V.row(i).transpose() = T.linear()*V.row(i).transpose() + T.translation();
  }

  // cleanup
  for(int j = 0; j < n; j++) {
    delete ap[j];
  }
  delete [] ap;
}

float trueSDF(Eigen::Vector3d &p) {
  QUERY_COUNTER ++;
  Eigen::MatrixXd q(1,3); 
  q.row(0) = p;

  Eigen::VectorXd S;
  Eigen::VectorXi I;
  Eigen::MatrixXd N,C;
  // Bunny is a watertight mesh so use pseudonormal for signing
  signed_distance_pseudonormal(q,V,F,tree,FN,VN,EN,EMAP,S,I,C,N);
  return float(S[0]);
}

float inferSDF(Eigen::Vector3d &p) {
  QUERY_COUNTER ++;
  Eigen::MatrixXf q(1,3);
  q.row(0) = p.cast <float> ();

  Eigen::VectorXf S = -mlp.predict(q);

  // just return as a float, not vector
  return S[0];
}

// nice implementation: https://www.shadertoy.com/view/4tcGDr
float shortestDistanceToSurface(Eigen::Vector3d &eye, Eigen::Vector3d &dir, float start, float end, float (*sdf)(Eigen::Vector3d &query)) {
    float depth = start;

    Eigen::Vector3d p; 

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
      p = eye + depth * dir;

      float dist = sdf(p);

      if (dist < EPSILON) {
        return depth;
      }
      depth += dist;
      if (depth >= end) {
          return end;
      }
    }
  
    return end;
}


// field of view in rads!
Eigen::Vector3d rayDirection(float fieldOfView, Eigen::Vector2i &size, Eigen::Vector2i &coord) {
    Eigen::Vector2d xy = (coord.cast <double> ()) - (size.cast <double> ()) / 2.0;
    float z = size[1] / std::tan(fieldOfView / 2.0);

    Eigen::Vector3d rayDir;
    rayDir << xy[0] , xy[1], z;

    rayDir.normalize();

    return rayDir;
}

Eigen::MatrixXd lookAt(Eigen::Vector3d eye, Eigen::Vector3d center, Eigen::Vector3d up) {
    Eigen::Vector3d f = (center - eye).normalized();
    Eigen::Vector3d u = up.normalized();
    Eigen::Vector3d s = f.cross(u).normalized();
    u = s.cross(f);
    
    Eigen::Matrix4d res;

    res << s.x(), s.y(), s.z(), -s.dot(eye),
            u.x(),u.y(),u.z(),-u.dot(eye),
            -f.x(),-f.y(),-f.z(),f.dot(eye),
            0,0,0,1;

    return res;
}

Eigen::Vector3d fragNormal(Eigen::Vector3d &p, float (*sdf)(Eigen::Vector3d &query)) {
  Eigen::Vector3d n;
  Eigen::Vector3d q1,q2;

  q1 = Eigen::Vector3d(p.x() + EPSILON, p.y(), p.z());
  q2 = Eigen::Vector3d(p.x() - EPSILON, p.y(), p.z());
  n[0] = sdf(q1) - sdf(q2);

  q1 = Eigen::Vector3d(p.x(), p.y() + EPSILON, p.z());
  q2 = Eigen::Vector3d(p.x(), p.y() - EPSILON, p.z());
  n[1] = sdf(q1) - sdf(q2);

  q1 = Eigen::Vector3d(p.x(), p.y(), p.z() + EPSILON);
  q2 = Eigen::Vector3d(p.x(), p.y(), p.z() - EPSILON);
  n[2] = sdf(q1) - sdf(q2);

  return n.normalized();
}

Eigen::Vector3d phongContribForLight(Eigen::Vector3d &k_d, Eigen::Vector3d &k_s, float alpha, 
                          Eigen::Vector3d &p, Eigen::Vector3d &eye,Eigen::Vector3d &lightPos, 
                          Eigen::Vector3d &lightIntensity, float (*sdf)(Eigen::Vector3d &query) ) {

    Eigen::Vector3d N = fragNormal(p, sdf);

    Eigen::Vector3d L = (lightPos - p).normalized();

    Eigen::Vector3d V = (eye - p).normalized();

    Eigen::Vector3d R = -L - 2.0 * N.dot(-L) * N;  // reflectance dir
    
    float dotLN = L.dot(N);

    float dotRV = R.dot(V);

    
    if (dotLN < 0.0) {
      // Light not visible from this point on the surface
      return Eigen::Vector3d(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
      // diffuse only
      return lightIntensity.cwiseProduct(k_d * dotLN);
    }


    return lightIntensity.cwiseProduct(k_d * dotLN + k_s * pow(dotRV, alpha)) ;
}

Eigen::Vector3d phongIllumination(Eigen::Vector3d &k_a, Eigen::Vector3d &k_d, Eigen::Vector3d &k_s, 
                          float alpha, Eigen::Vector3d &p, Eigen::Vector3d &eye, float (*sdf)(Eigen::Vector3d &query)) {
                            
    const Eigen::Vector3d ambientLight = 0.5 * Eigen::Vector3d(1.0, 1.0, 1.0);
    Eigen::Vector3d color = ambientLight.cwiseProduct(k_a);
    
    //Eigen::Vector3d light1Pos = Eigen::Vector3d(
    //                              5.0,
    //                              2.0,
    //                              5.0);
    //Eigen::Vector3d light1Intensity = Eigen::Vector3d(0.4, 0.4, 0.4);
    
    //color += phongContribForLight(k_d, k_s, alpha, p, eye,
    //                              light1Pos,
    //                              light1Intensity);

    
    Eigen::Vector3d light2Pos = eye;
    Eigen::Vector3d light2Intensity = Eigen::Vector3d(0.8, 0.8, 0.8);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity, sdf);    
    return color;
}

// Find t value for vector intersection with line
// P, start point of ray
// U, unit vector of ray
// C, center point of sphere
// r, radius of sphere
Eigen::Vector2d sphereIntersectionPoint(
  Eigen::Vector3d &P, Eigen::Vector3d &U, Eigen::Vector3d C, float r) {
    Eigen::Vector2d minMax; // the near and far intersections with sphere
    Eigen::Vector3d Q = P - C;
    double a = U.dot(U);
    double b = 2.0 * Q.dot(U);
    double c = Q.dot(Q) - r*r;
    double discriminant = b*b - 4*a*c;

    if(discriminant < 0){
      minMax << MAX_DIST, MAX_DIST;
    }
    else{
      minMax << (-b - sqrt(discriminant)) / (2.0*a), (-b + sqrt(discriminant)) / (2.0*a);
    }

    return minMax;
}

void triangleMeshLoader(const char * inputFilePath) {
  igl::read_triangle_mesh(inputFilePath, V, F);
  sphere_normalization(V, r);
  // Precompute signed distance AABB tree
  tree.init(V,F);
  // Precompute vertex,edge and face normals
  igl::per_face_normals(V,F,FN);
  igl::per_vertex_normals(V,F,igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,FN,VN);
  igl::per_edge_normals(V,F,igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,FN,EN,E,EMAP);
}

Eigen::Vector3i fragColor(
  Eigen::Vector2i &fragCoord, 
  Eigen::Vector2i &size, 
  Eigen::Vector3d &eye, 
  Eigen::MatrixXd &viewToWorld, 
  int shaderType,
  float r, 
  float (*sdf)(Eigen::Vector3d &query) ) 
  
{
  Eigen::Vector3i color;

  Eigen::Vector3d ray  = rayDirection(0.61, size, fragCoord);
  Eigen::Vector3d worldDir = viewToWorld * ray;

  // only start tracing within sphere
  Eigen::Vector2d nearFar = sphereIntersectionPoint(eye, worldDir, Eigen::Vector3d(0.0, 0.0, 0.0), r);

  float t = MAX_DIST;
  if (nearFar[0] < nearFar[1]){
    t = shortestDistanceToSurface(eye, worldDir, nearFar[0], nearFar[1], sdf);
  }

  if (t > nearFar[1] - EPSILON) {
    // didn't hit anything
    return Eigen::Vector3i(0,0,0);
  } 

  Eigen::Vector3d p = eye + t*worldDir;

  switch (shaderType)
  {
    case outline:
      // silllloutee (a word I can't spell)
      return Eigen::Vector3i(255,255,255);
    case gray: 
    {
      float maxDist = eye.norm() + r; //outside edge of bounding sphere
      float minDist = eye.norm() - r; // inside edge of bounding sphere

      int c = int(((abs(t) - minDist)/(maxDist-minDist))*255.0);

      return Eigen::Vector3i(c,c,c);
    }
    case phong: 
    {
      // phong
      Eigen::Vector3d K_a = (fragNormal(p,sdf) + Eigen::Vector3d(1.0, 1.0, 1.0)) / 2.0;
      Eigen::Vector3d K_d = K_a;
      Eigen::Vector3d K_s = Eigen::Vector3d(1.0, 1.0, 1.0);
      float shininess = 10.0;

      Eigen::Vector3d color = phongIllumination(K_a, K_d, K_s, shininess, p, eye, sdf);

      return Eigen::Vector3i(
        int(color[0]*255.999),
        int(color[1]*255.999),
        int(color[2]*255.999)
      );
    }
  }
}


char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[])
{
  // Defaults
  int H = 64;
  int W = 64;
  const char * inputFilePath = "bumpy-cube.obj";
  const char * outputFilePath = "bumpy-cube.png";
  int shaderType = outline;
  // sdf function pointer
  float (*pSDF)(Eigen::Vector3d &p){trueSDF};  // default to trueSDF

  // get cmd options
  if(cmdOptionExists(argv, argv+argc, "-H"))
  {
    H = atoi(getCmdOption(argv, argv + argc, "-H"));
  }

  if(cmdOptionExists(argv, argv+argc, "-W"))
  {
    W = atoi(getCmdOption(argv, argv + argc, "-W"));
  }

  if(cmdOptionExists(argv, argv+argc, "-i"))
  {
    inputFilePath = getCmdOption(argv, argv + argc, "-i");
    if (strstr(inputFilePath, ".h5")) {
      bool ok = mlp.load(inputFilePath);
      if (!ok) {
        std::cerr << "unable to load: " << inputFilePath << std::endl;
        return 0;
      }

      // point sdf to inference!
      pSDF = inferSDF;
    }
    // if not h5 must be mesh! (igl will throw if invalid)
    else {
      triangleMeshLoader(inputFilePath);
    }
  }

  if(cmdOptionExists(argv, argv+argc, "-o"))
  {
    outputFilePath = getCmdOption(argv, argv + argc, "-o");
  }

  if(cmdOptionExists(argv, argv + argc, "-s"))
  {
    shaderType = atoi(getCmdOption(argv, argv + argc, "-s"));
  }

  Eigen::Vector3d eye;
  eye << 0.0, 0.0, -5.0;

  Eigen::Vector2i size, coord;
  // image size
  size << H ,W;

  Eigen::MatrixXd viewToWorld = lookAt(eye, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.0, -1.0, 0.0));

  // our image channels
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(size[0],size[1]);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(size[0],size[1]);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(size[0],size[1]);
  Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(size[0],size[1]);

  std::clock_t startTime;
  startTime = std::clock();

  std::vector <Eigen::Vector2i> coords;

  for (int x = 0; x < size[0]; x ++) {
    for (int y = 0; y < size[1]; y ++) {
      Eigen::Vector2i coord(x,y);
      coords.push_back(coord);
    }
  }

  igl::parallel_for(coords.size(),[&](const int i)
  {
    std::cout << i << std::endl;
    Eigen::Vector2i coord = coords[i];
    Eigen::Vector3i RGB = fragColor(coord, size, eye, viewToWorld, shaderType, r, pSDF);

    R(coord[0],coord[1]) = RGB[0];
    G(coord[0],coord[1]) = RGB[1];
    B(coord[0],coord[1]) = RGB[2];
    A(coord[0],coord[1]) = 255;
  });

  std::cout <<  "Took: " << (std::clock() - startTime)/(double)(CLOCKS_PER_SEC / 1000) << " ms with "<< QUERY_COUNTER << " total queries\n";

  igl::png::writePNG(R,G,B,A, outputFilePath);

  return 1;
}

