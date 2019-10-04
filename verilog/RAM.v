module RAM_general(
input clk_in,
input clk_out,
input clk_in_im,
input clk_out_im,
input clk_in_im_scale,
input clk_out_im_scale,

input [15:0] data_in_im,
output reg [15:0] data_out_im,
input [18:0] address_in_im,
input [18:0] address_out_im,
input we_image,
input re_image,

input [23:0] data_in_im_scale,
output reg [23:0] data_out_im_scale,
input [15:0] address_in_im_scale,
input [15:0] address_out_im_scale,
input we_image_scale,
input re_image_scale,

input [20:0] data_in_weights,
output reg [20:0] data_out_weights,
input [23:0] address_in_weights,
input [23:0] address_out_weights,
input we_weights,
input re_weights

);

reg signed [15:0] ram_image [0:70800-1];
reg signed [23:0] ram_scale_image [0:128*128-1];
reg signed [20:0] ram_weights [0:(208115+2736)];

always @(posedge clk_in)	if (we_weights) ram_weights[address_in_weights] = data_in_weights;

always @(posedge clk_out)	if (re_weights) data_out_weights = ram_weights[address_out_weights];

always @(posedge clk_in_im) if (we_image)	ram_image[address_in_im] = data_in_im;

always @(posedge clk_out_im) if (re_image) data_out_im = ram_image[address_out_im];

always @(posedge clk_in_im_scale) if (we_image_scale)	ram_scale_image[address_in_im_scale] = data_in_im_scale;

always @(posedge clk_out_im_scale) if (re_image_scale) data_out_im_scale = ram_scale_image[address_out_im_scale];

endmodule
