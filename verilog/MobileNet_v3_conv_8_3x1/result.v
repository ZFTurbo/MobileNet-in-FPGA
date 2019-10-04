module result(clk,enable,STOP,memstartp,read_addressp,qp,re,RESULT);

parameter SIZE_1=0;
parameter SIZE_2=0;
parameter SIZE_3=0;
parameter SIZE_4=0;
parameter SIZE_5=0;
parameter SIZE_6=0;
parameter SIZE_7=0;
parameter SIZE_8=0;
parameter SIZE_address_pix=0;

input clk,enable;
output reg STOP;
input [SIZE_address_pix-1:0] memstartp;
input [SIZE_8-1:0] qp;
output reg re;
output reg [SIZE_address_pix-1:0] read_addressp;
output reg [1:0] RESULT;

reg [2:0] marker;
reg signed [SIZE_1-1:0] p1,p2;
always @(posedge clk)
begin
if (enable==1)
begin
re=1;
case (marker)
	0: 	begin
		read_addressp=memstartp+0;
		end
	1: 	begin
		end
	2: 	begin
		p1=qp[SIZE_8-1:SIZE_7];
		p2=qp[SIZE_7-1:SIZE_6];
		RESULT=0; 
		if (p2>=p1) RESULT=1; 
		else  RESULT=0; 
		STOP=1; 
		end
	default: $display("Check case result");
endcase
marker=marker+1;
end
else 
begin
re=0;
marker=0;
STOP=0;
end
end

endmodule
