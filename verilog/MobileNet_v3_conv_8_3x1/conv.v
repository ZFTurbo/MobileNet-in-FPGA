module conv(clk,Y1,prov,matrix,matrix2,i,up_perm,down_perm,p1,p2,p3,w1,w2,w3,conv_en,dense_en,stride_plus_prov);

parameter SIZE=0;
parameter SIZE_address_pix=18;
parameter SIZE_weights=0;

input clk;
output reg signed [32-1:0] Y1;
input [1:0] prov;
input [7:0] matrix;
input [14:0] matrix2;
input [14:0] i;
input up_perm,down_perm;
input signed [SIZE-1:0] p1,p2,p3;
input signed [SIZE_weights-1:0] w1,w2,w3;
input conv_en;
input dense_en;
input [SIZE_address_pix-1:0] stride_plus_prov;

wire up,down;

assign up = (((i+stride_plus_prov)<=matrix-1'b1)&&(up_perm))?1'b1:1'b0;
assign down = (((i+stride_plus_prov)>=matrix2-matrix)&&(down_perm))?1'b1:1'b0;

always @(posedge clk)
    begin
		if (conv_en==1)
			begin
				Y1=0;
				if ((prov!=2'b11)&&(!up)&&(!down)) Y1 = Y1+(p1*w1);
				if                ((!up)&&(!down)) Y1 = Y1+(p2*w2);
				if ((prov!=2'b10)&&(!up)&&(!down)) Y1 = Y1+(p3*w3);
			end
    end

endmodule
