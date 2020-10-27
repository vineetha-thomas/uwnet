#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix m: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
matrix forward_convolutional_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols % b.cols == 0);

    matrix y = copy_matrix(xw);
    int spatial = xw.cols / b.cols;
    int i,j;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            y.data[i*y.cols + j] += b.data[j/spatial];
        }
    }
    return y;
}

// Calculate bias updates from a delta matrix
// matrix delta: error made by the layer
// matrix db: delta for the biases
matrix backward_convolutional_bias(matrix dy, int n)
{
    assert(dy.cols % n == 0);
    matrix db = make_matrix(1, n);
    int spatial = dy.cols / n;
    int i,j;
    for(i = 0; i < dy.rows; ++i){
        for(j = 0; j < dy.cols; ++j){
            db.data[j/spatial] += dy.data[i*dy.cols + j];
        }
    }
    return db;
}

// Make a column matrix out of an image
// image im: image to process
// int size: kernel size for convolution operation
// int stride: stride for convolution
// returns: column matrix
matrix im2col(image im, int size, int stride)
{
    int i, j, k;
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;
    int cols = outw * outh;
    matrix col = make_matrix(rows, cols);

    // TODO: 5.1
    // Fill in the column matrix with patches from the image
    /* 
    //TRIAL1
    int im_x, im_y, im_c;
    for(i=0; i<rows; i++)
    {
        for(j=0; j<cols; j++)
        {
            im_x = -size/2 + (j/outw)*stride ;
            im_y = -size/2 + j%outw + i%size;
            im_c = i/(size*size);

            col.data[i*col.cols + j] = get_pixel(im, im_x, im_y, im_c);

        }
    }
    */

    // printf("im.w=%d \n", im.w);
    // printf("im.h=%d \n", im.h);
    // printf("size=%d \n", size);
    // printf("stride=%d \n", stride);
    // printf("matrix rows = %d \n", rows);
    // printf("matrix cols = %d \n", cols);  
    
    
    //TRIAL2
    /*
    int mr, mc;
    int l;
    int c;
    for(c=0; c<im.c; c++){
        for(i=0; i<im.h; i+=stride){
            for(j=0; j<im.w; j+=stride){
                mc = (i/stride)*outw + j/stride;

                mr = c*size*size;
                for(k=-size/2; k<size/2; k++){
                    for(l=-size/2; l<size/2; l++){
                        //printf("mr=%d \n", mr);
                        //printf("mc=%d \n", mc);
                        if (k+i >=0 && l+j>=0 && k+i<im.h && l+j<im.w)
                            col.data[mr*col.cols + mc] = get_pixel(im, k+i, l+j, c); 
                        else
                            col.data[mr*col.cols + mc] = 0;

                        mr+=1;
                    }
                }

            }
        }
        printf("col.data[34] = %f \n", col.data[34]);
    }
    */
    

    //TRIAL3
    for(k=0; k<rows; k++){
        int im_w_o = k%size;
        int im_h_o = (k/size)%size;
        int im_ch  = k/(size*size);

        for(i=0; i<outh; i++){ //out image height
            for(j=0; j<outw; j++){ //out image width
                int im_r = im_h_o + i*stride;
                int im_c = im_w_o + j*stride;

                im_r -= (size-1)/2;
                im_c -= (size-1)/2;

                if(im_r>=0 && im_r<im.h && im_c>=0 && im_c<im.w)
                    col.data[(k*outh + i)*outw + j] = get_pixel(im, im_c, im_r, im_ch);//im.data[(im_ch*im.h + im_r)*im.w + im_c];
                else    
                    col.data[(k*outh + i)*outw + j] = 0;
            }
        }
 
    }
  
    return col;
}

// The reverse of im2col, add elements back into image
// matrix col: column matrix to put back into image
// int size: kernel size
// int stride: convolution stride
// image im: image to add elements back into
image col2im(int width, int height, int channels, matrix col, int size, int stride)
{
    int i, j, k;

    image im = make_image(width, height, channels);
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;

    // TODO: 5.2
    // Add values into image im from the column matrix
     for(k=0; k<rows; k++){
        int im_w_o = k%size;
        int im_h_o = (k/size)%size;
        int im_ch  = k/(size*size);

        for(i=0; i<outh; i++){ //out image height
            for(j=0; j<outw; j++){ //out image width
                int im_r = im_h_o + i*stride;
                int im_c = im_w_o + j*stride;

                im_r -= (size-1)/2;
                im_c -= (size-1)/2;

                if(im_r>=0 && im_r<im.h && im_c>=0 && im_c<im.w)
                    im.data[(im_ch*im.h + im_r)*im.w + im_c] += col.data[(k*outh + i)*outw + j];
            }
        }
 
    }   


    return im;
}

// Run a convolutional layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_convolutional_layer(layer l, matrix in)
{
    assert(in.cols == l.width*l.height*l.channels);
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int i, j;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.filters);
    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        matrix wx = matmul(l.w, x);
        for(j = 0; j < wx.rows*wx.cols; ++j){
            out.data[i*out.cols + j] = wx.data[j];
        }
        free_matrix(x);
        free_matrix(wx);
    }
    matrix y = forward_convolutional_bias(out, l.b);
    free_matrix(out);

    return y;
}

// Run a convolutional layer backward
// layer l: layer to run
// matrix dy: derivative of loss wrt output dL/dy
matrix backward_convolutional_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    assert(in.cols == l.width*l.height*l.channels);

    int i;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;


    matrix db = backward_convolutional_bias(dy, l.db.cols);
    axpy_matrix(1, db, l.db);
    free_matrix(db);


    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);
    matrix wt = transpose_matrix(l.w);

    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);

        dy.rows = l.filters;
        dy.cols = outw*outh;

        matrix x = im2col(example, l.size, l.stride);
        matrix xt = transpose_matrix(x);
        matrix dw = matmul(dy, xt);
        axpy_matrix(1, dw, l.dw);

        matrix col = matmul(wt, dy);
        image dxi = col2im(l.width, l.height, l.channels, col, l.size, l.stride);
        memcpy(dx.data + i*dx.cols, dxi.data, dx.cols * sizeof(float));
        free_matrix(col);

        free_matrix(x);
        free_matrix(xt);
        free_matrix(dw);
        free_image(dxi);

        dy.data = dy.data + dy.rows*dy.cols;
    }
    free_matrix(wt);
    return dx;

}

// Update convolutional layer
// layer l: layer to update
// float rate: learning rate
// float momentum: momentum term
// float decay: l2 regularization term
void update_convolutional_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 5.3

    // Apply our updates using our SGD update rule
    // assume  l.dw = dL/dw - momentum * update_prev
    // we want l.dw = dL/dw - momentum * update_prev + decay * w
    // then we update l.w = l.w - rate * l.dw
    // lastly, l.dw is the negative update (-update) but for the next iteration
    // we want it to be (-momentum * update) so we just need to scale it a little

    axpy_matrix(decay, l.w, l.dw);
    axpy_matrix(-rate, l.dw, l.w);
    scal_matrix(momentum, l.dw);

    // Do the same for biases as well but no need to use weight decay on biases

    axpy_matrix(-rate, l.db, l.b);
    scal_matrix(momentum, l.db);
}

// Make a new convolutional layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of convolutional filter to apply
// int stride: stride of operation
layer make_convolutional_layer(int w, int h, int c, int filters, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.filters = filters;
    l.size = size;
    l.stride = stride;
    l.w  = random_matrix(filters, size*size*c, sqrtf(2.f/(size*size*c)));
    l.dw = make_matrix(filters, size*size*c);
    l.b  = make_matrix(1, filters);
    l.db = make_matrix(1, filters);
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update   = update_convolutional_layer;
    return l;
}

