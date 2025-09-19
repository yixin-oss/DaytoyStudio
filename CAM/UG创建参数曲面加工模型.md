## 问题需求

对于已知表达式的参数曲面$S(u,v)$, 结合MTALAB在UG中对其进行**建模**得到**工件**, 从而进入下一步的**仿真加工**环节. 

## 探索

首先UG不像MATLAB一样能根据表达式直接生成曲面，遂进行如下方案的尝试, 均有一定差强人意, 但还是要做个记录:

|                            Method                            |                         Pros & Cons                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|             利用MATLAB将曲面保存为STL文件导入UG              | 最直接的方案，统一文件的格式；然而UG只能将STL文件识别为收敛体，也就是说只显示图像但加工时无法选中 |
|           将STL文件利用网站转换为STEP文件再导入UG            | 转换过程涉及到逆向建模，虽然借助现有网站可以实现, 但得到的文件是大量分散的小曲面片, 需要再借助UG进行拼接合并等工作，耗时费力，且处理过程越多精度损失越大，20000个小面片不知道要处理到猴年马月，遂放弃 |
| 利用MATLAB生成参数曲面的离散点.dat文件, 然后UG生成曲面: 从文件中的点 | 点的分布要求极其规律如矩形阵列，可通过矩形点列成链的功能生成类似于平面的极简单曲面 |
|          仍然是离散点.dat文件, UG生成曲面: 拟合曲面          | 全选所有点，选择拟合方向为方位后调整参数化次数和补片数即可得到拟合结果, 操作简单结果易见, 缺点是拟合曲面并不是精确构造, 可能会在部分区域出现较大的拟合误差(UG会给出误差数值),或出现拟合结果扩大的情形, 并不完全适用需求 |

下面是每种尝试所需的步骤, 代码或资源

- 利用MATLAB将曲面保存为STL文件导入UG

Testufun.m->stlwrite.m->ParametricSurface.m

```matlab
function [aa,bb,cc,dd,x,y,z,ru,rv,ruu,ruv,rvv] = Testfun(u,v,ft)

%ft代表数值算例编号

%[aa,bb],[cc,dd]分别代表u,v取值范围
switch ft
    case 16 
        % 旋转抛物面(Rotating paraboloid)(旋转场)
        aa=-2;bb=2;cc=-2;dd=2;
        x=10*u;
        y=10*v;
        z=u.^2+v.^2;
        z=-10*z;
        ru=[10,0,-20*u];
        rv=[0,10,-20*v];
        ruu=[0,0,-20];
        ruv=[0,0,0];
        rvv=[0,0,-20];
        
   % case ...
        
end

end

```

```matlab
function stlwrite(filename, varargin)
%STLWRITE   Write STL file from patch or surface data.
%
%   STLWRITE(FILE, FV) writes a stereolithography (STL) file to FILE for a
%   triangulated patch defined by FV (a structure with fields 'vertices'
%   and 'faces').
%
%   STLWRITE(FILE, FACES, VERTICES) takes faces and vertices separately,
%   rather than in an FV struct
%
%   STLWRITE(FILE, X, Y, Z) creates an STL file from surface data in X, Y,
%   and Z. STLWRITE triangulates this gridded data into a triangulated
%   surface using triangulation options specified below. X, Y and Z can be
%   two-dimensional arrays with the same size. If X and Y are vectors with
%   length equal to SIZE(Z,2) and SIZE(Z,1), respectively, they are passed
%   through MESHGRID to create gridded data. If X or Y are scalar values,
%   they are used to specify the X and Y spacing between grid points.
%
%   STLWRITE(...,'PropertyName',VALUE,'PropertyName',VALUE,...) writes an
%   STL file using the following property values:
%
%   MODE          - File is written using 'binary' (default) or 'ascii'.
%
%   TITLE         - Header text (max 80 chars) written to the STL file.
%
%   TRIANGULATION - When used with gridded data, TRIANGULATION is either:
%                       'delaunay'  - (default) Delaunay triangulation of X, Y
%                       'f'         - Forward slash division of grid quads
%                       'b'         - Back slash division of quadrilaterals
%                       'x'         - Cross division of quadrilaterals
%                   Note that 'f', 'b', or 't' triangulations now use an
%                   inbuilt version of FEX entry 28327, "mesh2tri".
%
%   FACECOLOR     - Single colour (1-by-3) or one-colour-per-face (N-by-3) 
%                   vector of RGB colours, for face/vertex input. RGB range
%                   is 5 bits (0:31), stored in VisCAM/SolidView format
%                   (http://en.wikipedia.org/wiki/STL_(file_format)#Color_in_binary_STL)
%
%   Example 1:
%     % Write binary STL from face/vertex data
%     tmpvol = false(20,20,20);      % Empty voxel volume
%     tmpvol(8:12,8:12,5:15) = 1;    % Turn some voxels on
%     fv = isosurface(~tmpvol, 0.5); % Make patch w. faces "out"
%     stlwrite('test.stl',fv)        % Save to binary .stl
%
%   Example 2:
%     % Write ascii STL from gridded data
%     [X,Y] = deal(1:40);             % Create grid reference
%     Z = peaks(40);                  % Create grid height
%     stlwrite('test.stl',X,Y,Z,'mode','ascii')
%
%   Example 3:
%     % Write binary STL with coloured faces
%     cVals = fv.vertices(fv.faces(:,1),3); % Colour by Z height.
%     cLims = [min(cVals) max(cVals)];      % Transform height values
%     nCols = 255;  cMap = jet(nCols);      % onto an 8-bit colour map
%     fColsDbl = interp1(linspace(cLims(1),cLims(2),nCols),cMap,cVals); 
%     fCols8bit = fColsDbl*255; % Pass cols in 8bit (0-255) RGB triplets
%     stlwrite('testCol.stl',fv,'FaceColor',fCols8bit) 

%   Original idea adapted from surf2stl by Bill McDonald. Huge speed
%   improvements implemented by Oliver Woodford. Non-Delaunay triangulation
%   of quadrilateral surface courtesy of Kevin Moerman. FaceColor
%   implementation by Grant Lohsen.
%
%   Author: Sven Holcombe, 11-24-11


% Check valid filename path
path = fileparts(filename);
if ~isempty(path) && ~exist(path,'dir')
    error('Directory "%s" does not exist.',path);
end

% Get faces, vertices, and user-defined options for writing
[faces, vertices, options] = parseInputs(varargin{:});
asciiMode = strcmp( options.mode ,'ascii');

% Create the facets
facets = single(vertices');
facets = reshape(facets(:,faces'), 3, 3, []);

% Compute their normals
V1 = squeeze(facets(:,2,:) - facets(:,1,:));
V2 = squeeze(facets(:,3,:) - facets(:,1,:));
normals = V1([2 3 1],:) .* V2([3 1 2],:) - V2([2 3 1],:) .* V1([3 1 2],:);
clear V1 V2
normals = bsxfun(@times, normals, 1 ./ sqrt(sum(normals .* normals, 1)));
facets = cat(2, reshape(normals, 3, 1, []), facets);
clear normals

% Open the file for writing
permissions = {'w','wb+'};
fid = fopen(filename, permissions{asciiMode+1});
if (fid == -1)
    error('stlwrite:cannotWriteFile', 'Unable to write to %s', filename);
end

% Write the file contents
if asciiMode
    % Write HEADER
    fprintf(fid,'solid %s\r\n',options.title);
    % Write DATA
    fprintf(fid,[...
        'facet normal %.7E %.7E %.7E\r\n' ...
        'outer loop\r\n' ...
        'vertex %.7E %.7E %.7E\r\n' ...
        'vertex %.7E %.7E %.7E\r\n' ...
        'vertex %.7E %.7E %.7E\r\n' ...
        'endloop\r\n' ...
        'endfacet\r\n'], facets);
    % Write FOOTER
    fprintf(fid,'endsolid %s\r\n',options.title);
    
else % BINARY
    % Write HEADER
    fprintf(fid, '%-80s', options.title);             % Title
    fwrite(fid, size(facets, 3), 'uint32');           % Number of facets
    % Write DATA
    % Add one uint16(0) to the end of each facet using a typecasting trick
    facets = reshape(typecast(facets(:), 'uint16'), 12*2, []);
    % Set the last bit to 0 (default) or supplied RGB
    facets(end+1,:) = options.facecolor;
    fwrite(fid, facets, 'uint16');
end

% Close the file
fclose(fid);
fprintf('Wrote %d faces\n',size(faces, 1));


%% Input handling subfunctions
function [faces, vertices, options] = parseInputs(varargin)
% Determine input type
if isstruct(varargin{1}) % stlwrite('file', FVstruct, ...)
    if ~all(isfield(varargin{1},{'vertices','faces'}))
        error( 'Variable p must be a faces/vertices structure' );
    end
    faces = varargin{1}.faces;
    vertices = varargin{1}.vertices;
    options = parseOptions(varargin{2:end});
    
elseif isnumeric(varargin{1})
    firstNumInput = cellfun(@isnumeric,varargin);
    firstNumInput(find(~firstNumInput,1):end) = 0; % Only consider numerical input PRIOR to the first non-numeric
    numericInputCnt = nnz(firstNumInput);
    
    options = parseOptions(varargin{numericInputCnt+1:end});
    switch numericInputCnt
        case 3 % stlwrite('file', X, Y, Z, ...)
            % Extract the matrix Z
            Z = varargin{3};
            
            % Convert scalar XY to vectors
            ZsizeXY = fliplr(size(Z));
            for i = 1:2
                if isscalar(varargin{i})
                    varargin{i} = (0:ZsizeXY(i)-1) * varargin{i};
                end                    
            end
            
            % Extract X and Y
            if isequal(size(Z), size(varargin{1}), size(varargin{2}))
                % X,Y,Z were all provided as matrices
                [X,Y] = varargin{1:2};
            elseif numel(varargin{1})==ZsizeXY(1) && numel(varargin{2})==ZsizeXY(2)
                % Convert vector XY to meshgrid
                [X,Y] = meshgrid(varargin{1}, varargin{2});
            else
                error('stlwrite:badinput', 'Unable to resolve X and Y variables');
            end
            
            % Convert to faces/vertices
            if strcmp(options.triangulation,'delaunay')
                faces = delaunay(X,Y);
                vertices = [X(:) Y(:) Z(:)];
            else
                if ~exist('mesh2tri','file')
                    error('stlwrite:missing', '"mesh2tri" is required to convert X,Y,Z matrices to STL. It can be downloaded from:\n%s\n',...
                        'http://www.mathworks.com/matlabcentral/fileexchange/28327')
                end
                [faces, vertices] = mesh2tri(X, Y, Z, options.triangulation);
            end
            
        case 2 % stlwrite('file', FACES, VERTICES, ...)
            faces = varargin{1};
            vertices = varargin{2};
            
        otherwise
            error('stlwrite:badinput', 'Unable to resolve input types.');
    end
end

if size(faces,2)~=3
    errorMsg = {
        sprintf('The FACES input array should hold triangular faces (N x 3), but was detected as N x %d.',size(faces,2))
        'The STL format is for triangulated surfaces (i.e., surfaces made from 3-sided triangles).'
        'The Geom3d package (https://www.mathworks.com/matlabcentral/fileexchange/24484-geom3d) contains'
        'a "triangulateFaces" function which can be used convert your faces into triangles.'
        };
    error('stlwrite:nonTriangles', '%s\n',errorMsg{:})
end

if ~isempty(options.facecolor) % Handle colour preparation
    facecolor = uint16(options.facecolor);
    %Set the Valid Color bit (bit 15)
    c0 = bitshift(ones(size(faces,1),1,'uint16'),15);
    %Red color (10:15), Blue color (5:9), Green color (0:4)
    c0 = bitor(bitshift(bitand(2^6-1, facecolor(:,1)),10),c0);
    c0 = bitor(bitshift(bitand(2^11-1, facecolor(:,2)),5),c0);
    c0 = bitor(bitand(2^6-1, facecolor(:,3)),c0);
    options.facecolor = c0;    
else
    options.facecolor = 0;
end

function options = parseOptions(varargin)
IP = inputParser;
IP.addParamValue('mode', 'binary', @ischar)
IP.addParamValue('title', sprintf('Created by stlwrite.m %s',datestr(now)), @ischar);
IP.addParamValue('triangulation', 'delaunay', @ischar);
IP.addParamValue('facecolor',[], @isnumeric)
IP.addParamValue('facecolour',[], @isnumeric)
IP.parse(varargin{:});
options = IP.Results;
if ~isempty(options.facecolour)
    options.facecolor = options.facecolour;
end

function [F,V]=mesh2tri(X,Y,Z,tri_type)
% function [F,V]=mesh2tri(X,Y,Z,tri_type)
% 
% Available from http://www.mathworks.com/matlabcentral/fileexchange/28327
% Included here for convenience. Many thanks to Kevin Mattheus Moerman
% kevinmoerman@hotmail.com
% 15/07/2010
%------------------------------------------------------------------------

[J,I]=meshgrid(1:1:size(X,2)-1,1:1:size(X,1)-1);

switch tri_type
    case 'f'%Forward slash
        TRI_I=[I(:),I(:)+1,I(:)+1;  I(:),I(:),I(:)+1];
        TRI_J=[J(:),J(:)+1,J(:);   J(:),J(:)+1,J(:)+1];
        F = sub2ind(size(X),TRI_I,TRI_J);
    case 'b'%Back slash
        TRI_I=[I(:),I(:)+1,I(:);  I(:)+1,I(:)+1,I(:)];
        TRI_J=[J(:)+1,J(:),J(:);   J(:)+1,J(:),J(:)+1];
        F = sub2ind(size(X),TRI_I,TRI_J);
    case 'x'%Cross
        TRI_I=[I(:)+1,I(:);  I(:)+1,I(:)+1;  I(:),I(:)+1;    I(:),I(:)];
        TRI_J=[J(:),J(:);    J(:)+1,J(:);    J(:)+1,J(:)+1;  J(:),J(:)+1];
        IND=((numel(X)+1):numel(X)+prod(size(X)-1))';
        F = sub2ind(size(X),TRI_I,TRI_J);
        F(:,3)=repmat(IND,[4,1]);
        Fe_I=[I(:),I(:)+1,I(:)+1,I(:)]; Fe_J=[J(:),J(:),J(:)+1,J(:)+1];
        Fe = sub2ind(size(X),Fe_I,Fe_J);
        Xe=mean(X(Fe),2); Ye=mean(Y(Fe),2);  Ze=mean(Z(Fe),2);
        X=[X(:);Xe(:)]; Y=[Y(:);Ye(:)]; Z=[Z(:);Ze(:)];
end

V=[X(:),Y(:),Z(:)];
```

```matlab
ft=16;
% 初始化u,v取值范围
[aa,bb,cc,dd,~,~,~,~,~,~,~,~]=Testfun(0,0,ft);
% 设置采样点规模
u_grid=100;v_grid=100;
U=aa:(bb-aa)/u_grid:bb;
V=cc:(dd-cc)/v_grid:dd;
for i=1:length(V)
    for j=1:length(U)
        u=U(j);v=V(i);
        [~,~,~,~,x,y,z,~,~,~,~,~]=Testfun(u,v,ft);
        x1(i,j)=x;
        y1(i,j)=y;
        z1(i,j)=z;
    end
end
tri=delaunay(x1,y1);
% trimesh(tri,x1,y1,z1);
% xlabel('x');ylabel('y');zlabel('z')
mesh_data.vertices = [x1(:), y1(:), z1(:)];
mesh_data.faces = tri;

stlwrite('f16.stl', mesh_data);
```

- 将STL文件利用网站转换为STEP文件再导入UG

[stl2stp](http://stl2stp.cn/)

在线网站只支持1M以内的文件转换, 所以生成的STL文件要小一点.

- 利用MATLAB生成参数曲面的离散点.dat文件, 然后UG生成曲面: 从文件中的点

具体步骤可见如下CSDN博客

[UG NX 12 通过点构造面](https://jianhongwei1989.blog.csdn.net/article/details/119111469?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-4-119111469-blog-119112113.235%5Ev43%5Epc_blog_bottom_relevance_base9&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-4-119111469-blog-119112113.235%5Ev43%5Epc_blog_bottom_relevance_base9&utm_relevant_index=7)

- 仍然是离散点.dat文件, UG生成曲面: 拟合曲面

先生成离散点.dat文件

```matlab
ft=16;
% 初始化u,v取值范围
[aa,bb,cc,dd,~,~,~,~,~,~,~,~]=Testfun(0,0,ft);
% 设置采样点规模
u_grid=49;v_grid=49;
U=aa:(bb-aa)/u_grid:bb;
V=cc:(dd-cc)/v_grid:dd;
% 设计曲面可视化
% figure;
for i=1:length(V)
    for j=1:length(U)
        u=U(j);v=V(i);
        [~,~,~,~,x,y,z,~,~,~,~,~]=Testfun(u,v,ft);
        x1(i,j)=x;
        y1(i,j)=y;
        z1(i,j)=z;
    end
end
surf(x1,y1,z1);
hold on
shading interp

a=x1(:);
b=y1(:);
c=z1(:);
points=[a,b,c];

% 指定要保存的dat文件名
filename = 'f16_points_data.dat';

% 打开dat文件以进行写入
fileID = fopen(filename, 'w');

% 将点坐标数据写入dat文件
for i = 1:size(points, 1)
    fprintf(fileID, '%.6f %.6f %.6f\n', points(i, 1), points(i, 2), points(i, 3));
end

% 关闭dat文件
fclose(fileID);

disp('点坐标数据已成功保存为dat文件。');

```

该方案对于给出的旋转抛物面实例基本没问题, 但是对于以下马鞍面出现了拟合区域扩大的问题

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240722204327442.png" alt="image-20240722204327442" style="zoom:53%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240722204153479.png" alt="image-20240722204153479" style="zoom: 50%;" />



最后找到了一种可行的方案，原理也是最直接的“**点动成线，线动成面，面动成体**”, 果然"高端的食材往往只需要最简单的烹饪方式😅

以给出的旋转抛物面为例, 其UV方向本质都是抛物线, 因此可以根据导入的点, 利用UG生成"一般二次曲线"功能，每次选取5个点在两个方向上生成抛物线组, 再利用UG”通过曲线网格“功能, 分别选取主曲线(截线组)和作为引导曲线的交叉线生成曲面, 最后利用"加厚"功能, 通过在法向上选择加工尺寸得到最终的工件(可在加工模块中被选中).

- 应用模块: 建模, 插入曲线-> 一般二次曲线, 依次选取5个点->应用 得到一条曲线, 再选取下一组5个点逐步进行直至达到边界

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240722205259272.png" alt="image-20240722205259272" style="zoom:60%;" />

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240722205556640.png" alt="image-20240722205556640" style="zoom: 60%;" />

类似地, 在另一个方向上也先生成曲线, 对于上述简单实例, 只需在边界生成两条引导曲线即可.

- 插入->曲面->通过曲线网格

主曲线选取刚才在一个方向上生成的截线, 每选取一条按鼠标中键确认后选取下一条, 注意截线上出现的方向的一致性(双击转变方向), 直至将所有截线选中;

交叉曲线选取另一个方向的引导曲线(刚生成的两条边界线), 同样操作选取一条按鼠标中键确认后选取第二条, 此时就可预览生成的曲面结果, 若提示截线无效, 可将其单独取消选中再重新生成.

生成曲面后, 为了整洁显示, 可以将离散点隐藏, 对曲面选择隐藏父项(所有的曲线), 只显示曲面

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240722210218246.png" alt="image-20240722210218246" style="zoom:67%;" />

- 加厚

选择生成的曲面, 可以设置在法向的哪一侧进行偏置以及对应的尺寸, 即可得到可加工的工件.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/Img/image-20240722210450752.png" alt="image-20240722210450752" style="zoom: 67%;" />

## 注

对于其他可能遇到的更复杂的参数曲面表达, 如何在UG中生成可仿真加工的文件, 仍有待进一步探索.











