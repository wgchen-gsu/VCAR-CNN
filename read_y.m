function Y = read_y(filename, frameno, rows, cols)
headerlength = 0;  %The size of the file header.

im_size = rows*cols*3/2;

fp = fopen(filename,'rb','b');  % "Big-endian" byte order.
if (fp<0)
   error(['Cannot open ' filename '.']);
end

fseek(fp, 0, 'bof');
offset = headerlength + (frameno-1) * im_size;
status = fseek(fp, offset, 'bof');
if (status<0)        
    error(['Error in seeking image no: ' frameno '.']);   
end

gray_data = fread(fp, rows*cols, 'uchar');
temp_data = reshape(gray_data, [cols, rows]);
Y         = temp_data';

fclose(fp);
return