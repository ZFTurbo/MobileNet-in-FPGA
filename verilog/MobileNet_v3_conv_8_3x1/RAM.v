module RAM(qp,qtp,qw,dp,dtp,dw,write_addressp,read_addressp,write_addresstp,read_addresstp,write_addressw,read_addressw,we_p,we_tp,we_w,re_p,re_tp,re_w,clk,clk_RAM_w,q_bias,d_bias,we_bias,re_bias,write_address_bias,read_address_bias);
parameter picture_size=0;	
parameter SIZE_1=0;
parameter SIZE_2=0;
parameter SIZE_3=0;
parameter SIZE_4=0;
parameter SIZE_5=0;
parameter SIZE_6=0;
parameter SIZE_7=0;
parameter SIZE_8=0;
parameter SIZE_address_pix=13;
parameter SIZE_address_pix_t=12;
parameter SIZE_address_wei=13;
parameter SIZE_address_image=16;
parameter SIZE_weights=0;
parameter SIZE_bias=0;

output reg signed [SIZE_8-1:0] qp;       //read data
output reg signed [32*8-1:0] qtp;       //read data
output reg signed [SIZE_weights*9-1:0] qw;      //read weight
output reg signed [SIZE_bias-1:0] q_bias;
input signed [SIZE_1*8-1:0] dp;   //write data
input signed [32*8-1:0] dtp;   //write data
input signed [SIZE_weights*9-1:0] dw;   //write weight
input signed [SIZE_bias-1:0] d_bias;
input [SIZE_address_pix-1:0] write_addressp, read_addressp;
input [SIZE_address_pix_t-1:0] write_addresstp, read_addresstp;
input [SIZE_address_wei-1:0] write_addressw, read_addressw;
input [10:0] write_address_bias,read_address_bias;
input we_p;
input we_tp;
input we_w;
input we_bias;
input re_p;
input re_tp;
input re_w;
input re_bias;
input clk,clk_RAM_w;

reg signed [SIZE_1*8-1:0] mem [0:128*128*1+4096*2-1];
reg signed [32*8-1:0] mem_t [0:4096-1];
reg signed [SIZE_weights*9-1:0] weight [0:4095]; 
reg signed [SIZE_bias-1:0] mem_bias [0:256];
always @ (posedge clk) 
    begin
      if (we_p)  mem[write_addressp] <= dp;
		if (we_tp) mem_t[write_addresstp] <= dtp;
    end
always @ (posedge clk_RAM_w)
	begin
		if (we_w) weight[write_addressw] <= dw;
		if (we_bias) mem_bias[write_address_bias] <= d_bias;
	end
always @ (posedge clk)
    begin
      if (re_p) qp <= mem[read_addressp];
		if (re_tp)qtp <= mem_t[read_addresstp];
      if (re_w) qw <= weight[read_addressw];
		if (re_bias) q_bias <= mem_bias[read_address_bias];
    end

endmodule
