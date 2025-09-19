---
title: Tool path planning method for parametric surfaces based on piecewise Coons stream function reconstruction on hierarchical T-meshes
---

I am pleased to introduce my paper titled "Tool path planning method for parametric surfaces based on piecewise Coons stream function reconstruction on hierarchical T-meshes". This paper proposes a novel global tool path planning method based on piecewise Coons stream function reconstruction on adaptively hierarchical T-meshes. The work has been accepted by the Journal of Manufacturing Science and Engineering and was published online on June 23. 

To facilitate a quick understanding of the paper, the abstract are presented at the beginning.

# Abstract

Vector field-based tool path planning methods have been widely used for freeform surface machining, as they can effectively capture preferred feed directions that reflect the designers' machining intent. Among these methods, the stream function reconstruction algorithm stands out for its capability to generate tool paths with high machining efficiency. During the reconstruction process, there are two key problems that remain to be solved: controlling scallop height between adjacent paths to reduce redundant machining, and addressing the computational inefficiencies of the commonly used tensor product B-spline functions, which lack local refinement properties. To overcome these limitations, this paper proposes a novel global tool path planning method based on piecewise Coons stream function reconstruction on adaptively hierarchical T-meshes. This approach introduces an optimal stream function reconstruction model that integrates three optimization objectives: tool path alignment with the vector field, uniform distribution of scallop height between adjacent paths, and smoothness of the tool paths. Subsequently, an adaptively piecewise Coons function reconstruction algorithm is developed, utilizing the Coons interpolation for precise and efficient tool path generation. Experimental results validate the effectiveness of the proposed method.

**Keywords**: Parametric surface machining; tool path planning; vector field; hierarchical T-mesh; piecewise Coons stream function

------

Subsequently, I will introduce the content of this paper from four aspects: Introduction & Related work, Methodology, Experimental results, and Conclusion, to progressively illustrate the novelty of the proposed method.

# Introduction

- In computer-aided design (CAD) / computer-aided manufacturing (CAM) systems, tool path planning is essential for the machining process of freeform surfaces, as it directly affects machining quality and efficiency.

- Current tool path generation methods

  - Traditional tool path planning methods: iso-parametric, iso-planar, iso-scallop height

    <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/%E5%9B%BE%E7%89%871.jpg" alt="传统加工路径生成方法" style="zoom: 30%;" />

    :no_entry_sign: Path sequence generation method

    :no_entry_sign: Ignore the impact of the machine tool's physical performance

  - **Vector field-based tool path planning methods**

    - Vector field construction
      - Maximum machining strip width
      - Maximum material removal rate
      - Maximum kinematic performance
      - ......
    - Tool path generation
      - Initial path selection and offsetting
      - Surface segmentation
      - Stream function reconstruction
      - ......

    :white_check_mark: Reflect the designers’ machining intent

    :white_check_mark: Ensure machining efficiency

In short, Vector field-based tool path planning methods have become a hot topic.

# Related work

![image-20250715114904822](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715114904822.png)

A block diagram is provided to clearly illustrate the related work on vector field-based tool path planning methods with corresponding references listed for each approach. Among these methods, **stream function reconstruction is preferred for generating smooth tool paths that enhance machining performance**. The core mathematical problem in this approach is **constructing an optimal stream function based on vector fields**. Tensor product B-spline functions are commonly used to represent this stream function due to their flexibility, and the streamlines derived from the optimal stream function naturally serve as tool paths. In practical applications, this approach faces several problems that need to be studied, as follows:

##  Existing problems :exclamation:

- **Scallop height control between adjacent paths and smoothness of streamlines.**
  - Uniform distribution of scallop height between adjacent streamlines should be considered to **mitigate redundant machining** in some local regions. 
  - Adding energy related to smoothness of streamlines can effectively **avoid sharp corners**.
  - These constraints are supposed to be incorporated into the optimization process to obtain the optimal stream function. 

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715120359282.png" alt="image-20250715120359282" style="zoom: 80%;" />

- **Absence of local refinement properties of the tensor B-spline functions.**
  - The tensor product B-spline functions require **global refinement** of rectangular meshes to handle significant variations in complex vector fields, resulting in **redundant control coefficients**.
  - Piecewise Coons functions defined on hierarchical T-meshes provide a better alternative by supporting local refinement.

![image-20250715120407776](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715120407776.png)

# Methodology

​	To overcome the above limitations, this paper presents a novel global tool path planning method for freeform surfaces based on piecewise Coons stream function reconstruction on hierarchical T-meshes.

- An optimal stream function reconstruction model is introduced by integrating three key optimization objectives: tool path alignment with the vector field, uniform distribution of scallop height between adjacent paths, and smoothness of the tool paths into a functional energy minimization problem.
- An adaptively piecewise Coons function reconstruction algorithm with local refinement has been developed, utilizing the Coons interpolation for precise and efficient tool path generation.
- The proposed method effectively reduces the computational time required for stream function reconstruction compared to the B-spline methods, thereby improving computational efficiency.

​	Our method has been tested on several examples, with computer simulations and physical machining experiments conducted to verify its effectiveness compared with some traditional methods.

## Overview of the proposed method

An overview is provided to guide readers through the flow of the proposed method.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/%E5%9B%BE%E7%89%871.png" alt="图片1" style="zoom: 67%;" />

(a) Input a parametric surface, (b) The vector field corresponding to the surface in (a), (c) A locally refined hierarchical T-mesh obtained by solving the optimization model with an adaptive algorithm, (d) Piecewise Coons stream function constructed on the hierarchical T-mesh in (c), (e) Streamlines are extracted in the parameter domain by iso-level curves of the stream function in (d), (f) Global tool paths are generated by inversely mapping the streamlines in (e) onto the surface, (g) Simulation results of the tool paths in UG software, (h) The physical machining result.

## Optimal stream function reconstruction model

- Tool path optimization is transformed into **stream function optimization**.

- Three main objectives are considered simultaneously to construct the optimal stream function reconstruction model. 

  ![image-20250715135218175](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715135218175.png)

## Reconstruction of piecewise Coons stream functions based on adaptively hierarchical T-meshes

This section presents a piecewise Coons stream function reconstruction algorithm based on hierarchical T-meshes. Coons interpolation basis functions are defined on each cell of the T-mesh to ensure $C^1$ continuous between adjacent cells. For clarity, we first introduce two fundamental concepts: hierarchical T-meshes and Coons surfaces. Next, the reconstruction algorithm will be discussed in detail.

### Hierarchical T-meshes

A T-mesh is a modified rectangular mesh that allows T-junctions on a 2D plane. A T-junction is an endpoint of one edge that lies within the interior of another edge, as shown by the red stars. A crossing-vertex, the intersection of two interior edges, is shown by black circles, in the same manner as boundary vertices. As a result, the vertices of a T-mesh are divided into two kinds: basis vertices (boundary vertices and crossing vertices) and non-basis vertices (T-junctions).

A hierarchical T-mesh is a special type of T-mesh that has a natural level structure. It is defined recursively. The process of generating hierarchical T-meshes is illustrated below. We start from an initial T-mesh $\Gamma_0$ at level $0$ as shown in (a), the next level T-mesh $\Gamma_1$ is obtained by subdividing some cells in $\Gamma_0$ according to several predefined rules. Each cell is subdivided into $2\times 2$ uniform subcells, as shown in (b). Similarly, the T-mesh $\Gamma_2$ at level $2$ is obtained by refining $\Gamma_2$, as shown in (c).

![image-20250715135931583](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715135931583.png)

### Coons surfaces

Coons surfaces are constructed through the bi-cubic Hermite interpolation. For a given rectangular cell $\theta_{1}^{k}$ with domain $[a,b]\times[c,d]$, denote its four vertices by $\{V_{i}^k\}_{i=1}^4$. The Coons surface on cell $\theta_{1}^{k}$ is formulated by
$$
\scriptsize
\begin{equation}
		\label{15}
		\begin{aligned}
			P(u,~v)|_{\theta_{1}^k}  =  (F_0(\frac{u-a}{c-a}),~ (c-a)G_0(\frac{u-a}{c-a}),~(c-a)G_1(\frac{u-a}{c-a}),~F_1(\frac{u-a}{c-a}))\\
			\cdot
			\begin{pmatrix}
				p(V_1^k) & p_v(V_1^k) & p_v(V_4^k) & p(V_4^k)\\
				p_u(V_1^k) & p_{uv}(V_1^k) & p_{uv}(V_4^k) &p_u(V_4^k)\\
				p_u(V_2^k) & p_{uv}(V_2^k) & p_{uv}(V_3^k) &p_u(V_3^k)\\
				p(V_2^k) & p_v(V_2^k) & p_v(V_3^k) &p(V_3^k)
			\end{pmatrix}
			\cdot
			\begin{pmatrix}
				F_0(\frac{v-b}{d-b})\\
				(d-b)G_0(\frac{v-b}{d-b})\\
				(d-b)G_1(\frac{v-b}{d-b})\\
				F_1(\frac{t-b}{d-b})	
			\end{pmatrix}
			=\sum_{i=1}^{4}\sum_{j=1}^{4}h_{i,j}^kn_{i,j}^k(u,~v),		
		\end{aligned}
	\end{equation}
$$
where
$$
\begin{align}
		F_0(t)=1-3t^2+2t^3, G_0(t)=t(1-t)^2, G_1(t)=-t^2(1-t),F_1(t)=3t^2-2t^3,~ t\in[0,~1]
	\end{align}
$$
are cubic Hermite interpolation basis functions represented by the variable $t$. $(h_{1,1}^k,~ h_{1,2}^k,~...,~h_{4,4}^k)$ denote the 16 geometric parameters (function values, two first order partial derivatives, and mixed partial derivatives) at the four vertices of the cell. 

Based on the above two concepts, we **develop the piecewise Coons stream function reconstruction algorithm on locally refined hierarchical T-meshes**. To explain this process, we first discuss the simplest case, i.e., rectangular meshes, where all vertices are basis vertices used for the reconstruction of the stream function. For general T-meshes, a conversion matrix is applied to handle non-basis vertices, ensuring $C^1$ continuous between adjacent cells. Additionally, an adaptively refined stream function reconstruction algorithm is introduced to further improve computational efficiency.

- A rectangular mesh
- A general T-mesh → conversion matrix M
- An adaptively refined T-mesh 

:sparkle: **Key mathematical techniques**:

- Coons interpolation basis functions are defined on each cell in hierarchical T-meshes
- Maintaining C^1 continuous between adjacent cells in T-meshes
- Developing an adaptive algorithm for enhancing computational efficiency 

### Piecewise Coons stream function reconstruction on rectangular meshes

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715140631954.png" alt="image-20250715140631954" style="zoom:67%;" />

We provide an example to illustrate the reconstruction of the piecewise Coons stream function $\Psi$ on a rectangular mesh. For a given vector field as shown in (a), we can obtain a $4\times 4$ rectangular mesh and corresponding piecewise Coons stream function $\Psi$ as shown in (b) and (c). Finally, we extract the streamlines in the parameter domain $\Omega$ by iso-level curves of the stream function $\Psi$, as shown in (d).

![image-20250715140902707](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715140902707.png)

:white_check_mark: Similar to the Finite Element Method (FEM), the reconstruction of the stream function can be simplified to solve a sparse linear system of equations with control coefficients using a cell-based approach.

### Piecewise Coons stream function reconstruction over general T-meshes

- The geometric parameters of the Coons stream function at the non-basis vertices are determined by those of basis vertices.
- A global conversion matrix is derived to transform the geometric parameters at basis vertices into all vertices of the T-mesh.

$$
\boldsymbol{h}=\boldsymbol{M}\boldsymbol{d},~ \boldsymbol{d}=(\boldsymbol{M}^{T}(\boldsymbol{K}+\lambda\boldsymbol{L})\boldsymbol{M})^{-1}\boldsymbol{M}^T \boldsymbol{B}.
$$

### An adaptively piecewise Coons stream function reconstruction algorithm over hierarchical T-meshes

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715141629675.png" alt="image-20250715141629675"  />

:white_check_mark: The adaptive algorithm designed for the locally refined hierarchical T-meshes can further enhance computational efficiency.

# Experimental results

## Computation of the quality of the tool paths

- Basic parameters: surface finish, tool path length, path smoothness.

- Three scores $(S_1, S_2, S_3 )$ are derived to quantify the quality of tool paths.

- Higher scores indicate better path quality.

![image-20250715105918492](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715105918492.png)

These scores are utilized as evaluation metrics for comparing the path quality generated by different methods.

## Case studies

To illustrate the feasibility of the proposed tool path generation method, three cases related to the milling of parametric surface with a ball-end mill are studied. A $4\times 4$ rectangular mesh is selected as the initial T-mesh $\Gamma_0$, which is used to iteratively reconstruct the stream function. All calculations involved in the proposed method are carried out on a computing environment of MATLAB installed on a desktop with a 2.4 GHz Intel Gold 5218 processor, 64 GB of RAM, and Windows 11.

Four traditional tool path generation methods, including **the B-spline method, the contour-parallel method, the iso-scallop height method, and the iso-parametric method, are adopted for comparison to demonstrate the effectiveness of the proposed method**. The relevant machining parameters are as follows: a ball-end mill with radius $R_t=6$ mm for case 1 and $R_t=4$ mm for case 2 and case 3, with a scallop height threshold $h_0=0.1$ mm. All tool paths generated by different methods are simulated in the commercial CAM software UG. The computation times of the proposed method and the B-spline method are recorded to further demonstrate the high computational efficiency of the proposed method.

### Case 1: Saddle surface ($80\times 80\times 80$ mm)

![image-20250715142103466](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715142103466.png)

### Case 2: Rotating paraboloid surface ($60\times 60 \times 80$ mm)

![image-20250715142219481](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715142219481.png)

### Case 3: Peak-shaped surface ($50\times 50\times 110$ mm)

![image-20250715142304690](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715142304690.png)

:black_square_button: **Statistics of computational results. Bold values means that the values are better than others.**

![image-20250715142438437](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715142438437.png)

:black_square_button: **Comparison between different methods for three case studies. Bold values means that the values are better than others.**

![image-20250715142519917](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715142519917.png)

:white_check_mark: The proposed method demonstrates a significant improvement in computational efficiency compared to the B-spline method.

:white_check_mark: The proposed method can obtain better path smoothness than the B-spline method, and perform best in total path length and average scallop height.

## Physical experiments

- The top half of a molar tooth (downloaded from [GrabCAD](https://grabcad.com/library))

  <img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715142727020.png" alt="image-20250715142727020" style="zoom: 80%;" />

- Machining process

  ![image-20250715142909225](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715142909225.png)

- Experimental results

  ![image-20250715142940925](https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20250715142940925.png)

:white_check_mark: The proposed method generated smoother tool paths compared to the B-spline method.

:white_check_mark: The proposed method can significantly improve the machining efficiency compared to the contour-parallel method (5.3 minutes vs. 13.9 minutes).

# Conclusion

This paper proposed a novel tool path planning method for machining parametric surfaces based on piecewise Coons stream function reconstruction on hierarchical T-meshes. The main works are summarized as follows: 

- Presented an optimal stream function reconstruction model by integrate three optimization objectives into a functional energy minimization problem: tool path alignment with the vector field, uniform distribution of scallop height between adjacent paths, and smoothness of the tool paths. 
- Establishing an adaptively piecewise Coons function reconstruction algorithm to construct the optimal stream function for tool path generation. 

:white_check_mark: The proposed method effectively improve the computational efficiency during the reconstruction process, compared to the B-spline method. 

:white_check_mark: The proposed method generates global tool paths that achieve both machining quality and efficiency compared to some existing tool path generation methods. 

# Acknowledgements

This work was supported by the National Key Research and Development Program of China (No. 2020YFA0713702), the National Natural Science Foundation of China (No. 12271079, 12494552), and the Fundamental Research Funds for the Central Universities of China (No. DUT24LAB127).

# Citation

Yingshi Li, Chongjun Li, Jiao Huang, Guangwen Yan, Jinting Xu, Ke Liu. Tool path planning method for parametric surfaces based on piecewise Coons stream function reconstruction over hierarchical T-meshes. ***Journal of Manufacturing Science and Engineering***, 2025, 147(9): 091008. https://doi.org/10.1115/1.4069010