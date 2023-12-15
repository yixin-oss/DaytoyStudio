---
title: MATLAB中3D模型的可视化

---

由于计算机辅助制造(CAM)中对刀具路径的规划需要在实体模型上进行仿真以保证路径规划算法的准确性和有效性, 那么在MATLAB的编程语言环境下, 一个最基本的问题是将实体模型的数据导入到MATLAB中进行可视化, 并且便于后续的数据处理。本文将从**模型获取→文件格式转换→MATLAB可视化**三方面进行介绍.

# 模型网站

[grabCAD](https://grabcad.com/library)

超实用的3D模型零件库，拥有400w+的模型，主要以机械加工类模型为主，并且**全部免费**。对网站进行**注册**后就可以下载模型了.

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20230918200543374.png" alt="image-20230918200543374"  />

[traceparts](http://traceparts.com/zh)

拥有数以百万计的 3D 模型、2D 图纸和 CAD 文件，并且也是**全部免费**。网站支持中文，可按需求自选不同的文件类型进行下载。

![image-20230918201020375](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20230918201020375.png)



[free3D](https://free3d.com/zh/)

缺点：模型数量10w+，相对较少，部分模型需要付费。

优点：免费的资源可直接下载。网站支持中文，界面简洁友好，可以按不同的类别寻找自己喜欢的模型，对各模型还会有规格的说明，使用非常方便。

![image-20230918203554747](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20230918203554747.png)

![image-20230918203525254](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20230918203525254.png)

[turbosquid](http://turbosquid.com)

专业的3D模型网站，主要以现实物体的3D模型为主，模型种类很丰富，同样分为免费和付费两种类型。网站需要**注册**才能下载模型。

![image-20230918210158482](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20230918210158482.png)



以上4个网站从模型丰富程度，免费下载程度，模型介绍详细程度，网站注册繁琐程度，响应速度，界面友好程度等各指标来评估，更推荐使用[grabCAD](https://grabcad.com/library)和[free3D](https://free3d.com/zh/)，不过也可以按照对各类模型的不同需求选择另外两个网站，适合自己的才是最好的。更多的模型网站还有待进一步的探索。



# STL文件转换

从网站上获取到的模型的文件类型是各种各样的，比如step, iges, 3ds, obj等，它们是工程工业设计中常见的3D文件格式。为了便于后续导入到MATLAB中, 需要将模型转换成统一的stl文件格式，这里用到了一个在线网站[魔猴云](http://www.mohou.com/tools/stlconverter.html )

可以将模型直接**拖拽**或者**点击上传选择文件**，等待转换完成后就可以下载对应的stl文件了。通常转换完成后，网站还会自动出现模型的3D预览界面。

![image-20230918211325347](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20230918211325347.png)

<img src="https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20230918211420648.png" alt="blade" style="zoom:80%;" />

# stl文件导入MATLAB

进入到最后一个环节，将得到的stl文件导入到MATLAB中，完成模型的可视化及为后续的数据处理做准备。这一步需要用到**stlread**函数, 导入的stl文件是三角网格曲面的形式，还需要借助**patch**函数显示模型。

```matlab
[F,V,N]=stlread(filename) % 从由文件名指示的STL文件中导入三角形面, 并分别返回面F, 顶点V和法向量N.
% 面和顶点按patch绘图函数使用的格式排列
```

**stlread**函数需要从MathWorks官方网站上进行下载(需要先注册MathWorks，并且要许可证)

![image-20230918214652926](https://gitee.com/yixin-oss/blogImage/raw/master/img/image-20230918214652926.png)

这里我们引用了Eric Johnson的工作(引用格式: Eric Johnson (2023). STL File Reader (https://www.mathworks.com/matlabcentral/fileexchange/22409-stl-file-reader), MATLAB Central File Exchange.)

```matlab
function varargout = stlread(file)
% STLREAD imports geometry from an STL file into MATLAB.
%    FV = STLREAD(FILENAME) imports triangular faces from the ASCII or binary
%    STL file idicated by FILENAME, and returns the patch struct FV, with fields
%    'faces' and 'vertices'.
%
%    [F,V] = STLREAD(FILENAME) returns the faces F and vertices V separately.
%
%    [F,V,N] = STLREAD(FILENAME) also returns the face normal vectors.
%
%    The faces and vertices are arranged in the format used by the PATCH plot
%    object.

% Copyright 2011 The MathWorks, Inc.

    if ~exist(file,'file')
        error(['File ''%s'' not found. If the file is not on MATLAB''s path' ...
               ', be sure to specify the full path to the file.'], file);
    end
    
    fid = fopen(file,'r');    
    if ~isempty(ferror(fid))
        error(lasterror); %#ok
    end
    
    M = fread(fid,inf,'uint8=>uint8');
    fclose(fid);
    
    [f,v,n] = stlbinary(M);
    %if( isbinary(M) ) % This may not be a reliable test
    %    [f,v,n] = stlbinary(M);
    %else
    %    [f,v,n] = stlascii(M);
    %end
    
    varargout = cell(1,nargout);
    switch nargout        
        case 2
            varargout{1} = f;
            varargout{2} = v;
        case 3
            varargout{1} = f;
            varargout{2} = v;
            varargout{3} = n;
        otherwise
            varargout{1} = struct('faces',f,'vertices',v);
    end

end


function [F,V,N] = stlbinary(M)

    F = [];
    V = [];
    N = [];
    
    if length(M) < 84
        error('MATLAB:stlread:incorrectFormat', ...
              'Incomplete header information in binary STL file.');
    end
    
    % Bytes 81-84 are an unsigned 32-bit integer specifying the number of faces
    % that follow.
    numFaces = typecast(M(81:84),'uint32');
    %numFaces = double(numFaces);
    if numFaces == 0
        warning('MATLAB:stlread:nodata','No data in STL file.');
        return
    end
    
    T = M(85:end);
    F = NaN(numFaces,3);
    V = NaN(3*numFaces,3);
    N = NaN(numFaces,3);
    
    numRead = 0;
    while numRead < numFaces
        % Each facet is 50 bytes
        %  - Three single precision values specifying the face normal vector
        %  - Three single precision values specifying the first vertex (XYZ)
        %  - Three single precision values specifying the second vertex (XYZ)
        %  - Three single precision values specifying the third vertex (XYZ)
        %  - Two unused bytes
        i1    = 50 * numRead + 1;
        i2    = i1 + 50 - 1;
        facet = T(i1:i2)';
        
        n  = typecast(facet(1:12),'single');
        v1 = typecast(facet(13:24),'single');
        v2 = typecast(facet(25:36),'single');
        v3 = typecast(facet(37:48),'single');
        
        n = double(n);
        v = double([v1; v2; v3]);
        
        % Figure out where to fit these new vertices, and the face, in the
        % larger F and V collections.        
        fInd  = numRead + 1;        
        vInd1 = 3 * (fInd - 1) + 1;
        vInd2 = vInd1 + 3 - 1;
        
        V(vInd1:vInd2,:) = v;
        F(fInd,:)        = vInd1:vInd2;
        N(fInd,:)        = n;
        
        numRead = numRead + 1;
    end
    
end


function [F,V,N] = stlascii(M)
    warning('MATLAB:stlread:ascii','ASCII STL files currently not supported.');
    F = [];
    V = [];
    N = [];
end

% TODO: Change the testing criteria! Some binary STL files still begin with
% 'solid'.
function tf = isbinary(A)
% ISBINARY uses the first line of an STL file to identify its format.
    if isempty(A) || length(A) < 5
        error('MATLAB:stlread:incorrectFormat', ...
              'File does not appear to be an ASCII or binary STL file.');
    end    
    if strcmpi('solid',char(A(1:5)'))
        tf = false; % ASCII
    else
        tf = true;  % Binary
    end
end
```

# Demo

下面是将stl文件导入到MATLAB并利用patch函数进行可视化的一个demo, 这里我用的是小猫模型, 可以任意替换自己需求的stl文件. Demo的文件资源在[这里](https://download.csdn.net/download/yixon_oss/88354552)

```matlab
%% 3D Model Demo
% This is short demo that loads and renders a 3D model of a human femur. It
% showcases some of MATLAB's advanced graphics features, including lighting and
% specular reflectance.

% Copyright 2011 The MathWorks, Inc.


%% Load STL mesh
% Stereolithography (STL) files are a common format for storing mesh data. STL
% meshes are simply a collection of triangular faces. This type of model is very
% suitable for use with MATLAB's PATCH graphics object.

% Import an STL mesh, returning a PATCH-compatible face-vertex structure
fv = stlread('Cat.stl');


%% Render
% The model is rendered with a PATCH graphics object. We also add some dynamic
% lighting, and adjust the material properties to change the specular
% highlighting.

patch(fv,'FaceColor',       [0.8 0.8 1.0], ...
         'EdgeColor',       'none',        ...
         'FaceLighting',    'gouraud',     ...
         'AmbientStrength', 0.15);

% Add a camera light, and tone down the specular highlighting
camlight('headlight');
material('dull');

% Fix the axes scaling, and set a nice view angle
axis('image');
view([-135 35]);
```

![Cat](https://gitee.com/yixin-oss/blogImage/raw/master/img/Cat.png)



