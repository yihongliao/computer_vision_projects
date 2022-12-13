for i=1:20
    fileName = sprintf('000 (%d).jpg', i);
    I0 = imread(fileName);
    imwrite(I0(1:6:end, 1:6:end, :), fileName);
end