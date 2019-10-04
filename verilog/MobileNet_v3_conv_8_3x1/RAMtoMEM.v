module memorywork(clk_RAM_w,data,data_bias,address,we_w,re_weights,re_bias,nextstep,dw,addrw,step_out,GO,in_dense,load_weights,onexone,address_bias,d_bias,load_bias,we_bias,write_address_bias);

parameter num_conv=0;

parameter SIZE_1=0;
parameter SIZE_2=0;
parameter SIZE_3=0;
parameter SIZE_4=0;
parameter SIZE_5=0;
parameter SIZE_6=0;
parameter SIZE_7=0;
parameter SIZE_8=0;
parameter SIZE_address_pix=0;
parameter SIZE_address_wei=0;
parameter SIZE_weights=0;
parameter SIZE_bias=0;

input clk_RAM_w;
input signed [SIZE_weights-1:0] data;
input signed [SIZE_bias-1:0] data_bias;
output [23:0] address;
output reg we_w;
output re_weights,re_bias;
input nextstep;
output reg signed [SIZE_weights*9-1:0] dw;
output reg [SIZE_address_wei-1:0] addrw;
output [6:0] step_out;
input GO;
input [8:0] in_dense;
input load_weights,load_bias;

output reg signed [SIZE_bias-1:0] d_bias;
output reg we_bias;
output reg [10:0] write_address_bias;
output [11:0] address_bias;

input onexone; 

reg [SIZE_address_pix-1:0] addr;
wire [17:0] firstaddr,lastaddr;

wire [18:0] razn_addr;
assign razn_addr = lastaddr-firstaddr;

reg [6:0] step;
reg [6:0] step_n;
reg [3:0] weight_case;
reg [SIZE_weights*9-1:0] buff;
reg [17:0] i;
reg [17:0] i_d;
reg [17:0] i1;
addressRAM inst_1(.step(step_out),.re_weights(re_weights),.re_bias(re_bias),.firstaddr(firstaddr),.lastaddr(lastaddr));  
initial weight_case=0;
initial i=0;
initial i_d=0;
initial i1=0;

always @(negedge clk_RAM_w)
	if (  (step_out==1)||(step_out==2)
		||(step_out==4)||(step_out==5)
		||(step_out==7)||(step_out==8)
		||(step_out==10)||(step_out==11)
		||(step_out==13)||(step_out==14)
		||(step_out==16)||(step_out==17)
		||(step_out==19)||(step_out==20)
		||(step_out==22)||(step_out==23)
		||(step_out==25)||(step_out==26)
		||(step_out==28)||(step_out==29)
		||(step_out==31)||(step_out==32)
		||(step_out==34)||(step_out==35)
		||(step_out==37)||(step_out==38)
		||(step_out==40)||(step_out==41)
		||(step_out==43)||(step_out==44)
		||(step_out==46)||(step_out==47)
		||(step_out==49)||(step_out==50)
		||(step_out==52)||(step_out==53)
		||(step_out==55)||(step_out==56)
		||(step_out==58)||(step_out==59)
		||(step_out==61)||(step_out==62)
		||(step_out==64)||(step_out==65)
		||(step_out==67)||(step_out==68)
		||(step_out==70)||(step_out==71)
		||(step_out==73)||(step_out==74)
		||(step_out==76)||(step_out==77)
		||(step_out==79)||(step_out==80)
		||(step_out==82)||(step_out==83)
		||(step_out==85)
		)
	begin
		if ((i<=razn_addr+1)&&(re_weights))  addr=i1;
		if ((i<=razn_addr+1)&&(re_bias))	addr=i;
	end

always @(posedge clk_RAM_w or posedge GO)
	if (GO) step=1;
	else
    begin
			case (step_out)
				8'd1,8'd4,8'd7,8'd10,8'd13,8'd16,8'd19,8'd22,8'd25,8'd28,8'd31,8'd34,8'd37,8'd40,8'd43,8'd46,8'd49,8'd52,8'd55,8'd58,8'd61,8'd64,8'd67,8'd70,8'd73,8'd76,8'd79,8'd82,8'd85:
					begin
						if (i<=razn_addr+3)
                    begin
										we_w=0;
										addrw=addr;
										if (load_weights==1'b1) i=i+1; 
										if (step_out==85) if (i_d==((in_dense)+1)) begin  dw=buff; we_w=1; weight_case=1; i_d=1; i1=i1+1; end
										case (weight_case)
											0: ;
											1: begin buff=0; buff[SIZE_weights*9-1:SIZE_weights*8]=data[SIZE_weights-1:0]; end 
											2: buff[SIZE_weights*8-1:SIZE_weights*7]=data[SIZE_weights-1:0]; 
											3: buff[SIZE_weights*7-1:SIZE_weights*6]=data[SIZE_weights-1:0];  
											4: buff[SIZE_weights*6-1:SIZE_weights*5]=data[SIZE_weights-1:0];  
											5: buff[SIZE_weights*5-1:SIZE_weights*4]=data[SIZE_weights-1:0];  
											6: buff[SIZE_weights*4-1:SIZE_weights*3]=data[SIZE_weights-1:0]; 
											7: buff[SIZE_weights*3-1:SIZE_weights*2]=data[SIZE_weights-1:0]; 
											8: buff[SIZE_weights*2-1:SIZE_weights]=data[SIZE_weights-1:0];   
											9: begin buff[SIZE_weights-1:0]=data[SIZE_weights-1:0]; end
											default: $display("Check weight_case");
										endcase
										if (load_weights==1'b1) i_d=i_d+1;
										if (load_weights==1'b1)
											begin
												if ((weight_case==9)||((onexone)&&(weight_case==8))) 
													begin 
														weight_case=1; 
														dw=buff; 
														we_w=1; 
														i1=i1+1;
													end 
												else 
													begin
														weight_case=weight_case+1;
													end
										end
                    end
					if (i>razn_addr+3)
                    begin
                        step=step+1;          //next step
                        i=0;
						i_d=0;
						i1=0;
						weight_case=0;
                    end
            end
			8'd2,8'd5,8'd8,8'd11,8'd14,8'd17,8'd20,8'd23,8'd26,8'd29,8'd32,8'd35,8'd38,8'd41,8'd44,8'd47,8'd50,8'd53,8'd56,8'd59,8'd62,8'd65,8'd68,8'd71,8'd74,8'd77,8'd80,8'd83:
				begin
					if (i<=razn_addr)
						begin
							we_bias=1;
							write_address_bias=addr;
							if (load_bias==1'b1) i=i+1;
							d_bias=data_bias;
						end
					else	
						begin
							step=step+1;
							i=0;
							we_bias=0;
						end
				end
			default:
				begin
					we_w=0;
					we_bias=0;
					i=0;
					i_d=0;
					i1=0;
				end
		endcase
    end
always @(posedge nextstep) if (GO==1) step_n=0; else step_n=step_n+1;
assign step_out=step+step_n;
assign address=(re_weights)?(firstaddr+i):0;
assign address_bias=(re_bias)?(firstaddr+i):0;
endmodule
