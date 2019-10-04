module conv_TOP(clk,conv_en,STOP,memstartp,memstartw,memstartzap,read_addressp,write_addressp,read_addresstp,write_addresstp,read_addressw,we,re_wb,re,we_t,re_t,qp,qtp,qw,dp,dtp,prov,matrix,matrix2,i_to_prov,lvl,slvl,mem,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,w11,w12,w13,w21,w22,w23,w31,w32,w33,w41,w42,w43,w51,w52,w53,w61,w62,w63,w71,w72,w73,w81,w82,w83,p0_1,p0_2,p0_3,p1_1,p1_2,p1_3,p2_1,p2_2,p2_3,p3_1,p3_2,p3_3,p4_1,p4_2,p4_3,p5_1,p5_2,p5_3,p6_1,p6_2,p6_3,p7_1,p7_2,p7_3,go,up_perm,down_perm,num,filt,bias,glob_average_en,step,stride,depthwise,onexone,q_bias,read_addressb,memstartb,stride_plus_prov);

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
parameter SIZE_weights=0;
parameter SIZE_bias=0;

input clk,conv_en,glob_average_en;
input [1:0] prov;
input [7:0] matrix;
input [14:0] matrix2;
input [SIZE_address_pix-1:0] memstartp;
input [SIZE_address_wei-1:0] memstartw;
input [SIZE_address_pix-1:0] memstartzap;
input [10:0]				 memstartb;
input [8:0] lvl;
input [8:0] slvl;
output reg [SIZE_address_pix-1:0] read_addressp;
output reg [SIZE_address_pix_t-1:0] read_addresstp;
output reg [SIZE_address_wei-1:0] read_addressw;
output reg [10:0]				  read_addressb;
output reg [SIZE_address_pix-1:0] write_addressp;
output reg [SIZE_address_pix_t-1:0] write_addresstp;
output reg we,re,re_wb;
output reg we_t,re_t;
input signed [SIZE_8-1:0] qp;
input signed [32*8-1:0] qtp;
input signed [SIZE_weights*9-1:0] qw;
input signed [SIZE_bias-1:0] q_bias;
output signed [SIZE_8-1:0] dp;
output signed [32*8-1:0] dtp;
output reg STOP;
output reg [14:0] i_to_prov;
input signed [32-1:0] Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8;
output reg signed [SIZE_weights-1:0] w11,w12,w13,w21,w22,w23,w31,w32,w33,w41,w42,w43,w51,w52,w53,w61,w62,w63,w71,w72,w73,w81,w82,w83;
output reg signed [SIZE_1-1:0] p0_1,p0_2,p0_3,p1_1,p1_2,p1_3,p2_1,p2_2,p2_3,p3_1,p3_2,p3_3,p4_1,p4_2,p4_3,p5_1,p5_2,p5_3,p6_1,p6_2,p6_3,p7_1,p7_2,p7_3;
output reg go;
output reg up_perm,down_perm;
input [2:0] num;
input [8:0] mem;
input [8:0] filt;
input bias;
input [6:0] step;
input [1:0] stride;
output reg [SIZE_address_pix-1:0] stride_plus_prov;

input depthwise,onexone;

reg signed [SIZE_weights-1:0] w11_pre,w12_pre,w13_pre,w14_pre,w15_pre,w16_pre,w17_pre,w18_pre,w19_pre;
reg signed [SIZE_weights-1:0] w21_pre,w22_pre,w23_pre,w24_pre,w25_pre,w26_pre,w27_pre,w28_pre,w29_pre;
reg signed [SIZE_weights-1:0] w31_pre,w32_pre,w33_pre,w34_pre,w35_pre,w36_pre,w37_pre,w38_pre,w39_pre;
reg signed [SIZE_weights-1:0] w41_pre,w42_pre,w43_pre,w44_pre,w45_pre,w46_pre,w47_pre,w48_pre,w49_pre;
reg signed [SIZE_weights-1:0] w51_pre,w52_pre,w53_pre,w54_pre,w55_pre,w56_pre,w57_pre,w58_pre,w59_pre;
reg signed [SIZE_weights-1:0] w61_pre,w62_pre,w63_pre,w64_pre,w65_pre,w66_pre,w67_pre,w68_pre,w69_pre;
reg signed [SIZE_weights-1:0] w71_pre,w72_pre,w73_pre,w74_pre,w75_pre,w76_pre,w77_pre,w78_pre,w79_pre;
reg signed [SIZE_weights-1:0] w81_pre,w82_pre,w83_pre,w84_pre,w85_pre,w86_pre,w87_pre,w88_pre,w89_pre;
reg signed [SIZE_1-1:0]p0_pre,p1_pre,p2_pre,p3_pre,p4_pre,p5_pre,p6_pre,p7_pre,p8_pre,p9_pre,p10_pre,p11_pre,p12_pre,p13_pre,p14_pre,p15_pre;
reg signed [SIZE_1-1:0] res_out_1,res_out_2,res_out_3,res_out_4,res_out_5,res_out_6,res_out_7,res_out_8;
reg signed [32-1:0] res1,res2,res3,res4,res5,res6,res7,res8;
reg signed [32-1:0] res_old_1,res_old_2,res_old_3,res_old_4,res_old_5,res_old_6,res_old_7,res_old_8;
reg signed [21:0] glob_average_perem_1,glob_average_perem_2,glob_average_perem_3,glob_average_perem_4,glob_average_perem_5,glob_average_perem_6,glob_average_perem_7,glob_average_perem_8;
wire signed [SIZE_1-1:0] glob_average_perem_1_1,glob_average_perem_2_1,glob_average_perem_3_1,glob_average_perem_4_1,glob_average_perem_5_1,glob_average_perem_6_1,glob_average_perem_7_1,glob_average_perem_8_1;

reg signed [SIZE_1-1:0]buff0_0 [2:0], buff1_0 [2:0], buff2_0 [2:0], buff3_0 [2:0], buff4_0 [2:0], buff5_0 [2:0], buff6_0 [2:0], buff7_0 [2:0];
reg signed [SIZE_1-1:0]buff0_1 [2:0], buff1_1 [2:0], buff2_1 [2:0], buff3_1 [2:0], buff4_1 [2:0], buff5_1 [2:0], buff6_1 [2:0], buff7_1 [2:0];
reg signed [SIZE_1-1:0]buff0_2 [2:0], buff1_2 [2:0], buff2_2 [2:0], buff3_2 [2:0], buff4_2 [2:0], buff5_2 [2:0], buff6_2 [2:0], buff7_2 [2:0];

reg [4:0] marker;
reg zagryzka_weight;
reg [15:0] i;
reg [15:0] i_onexone,i_onexone_1;
wire [15:0] i_onexone_plus1;
assign i_onexone_plus1 = i_onexone + 1'b1;
reg [SIZE_address_pix-1:0] stride_plus,next_number,next_number_prov;

reg signed [19-1:0] res_bias_check_1,res_bias_check_2,res_bias_check_3,res_bias_check_4,res_bias_check_5,res_bias_check_6,res_bias_check_7,res_bias_check_8;

reg signed [SIZE_bias-1:0] data_bias_1,data_bias_2,data_bias_3,data_bias_4,data_bias_5,data_bias_6,data_bias_7,data_bias_8;

initial zagryzka_weight=0;
initial marker=0;

wire [15:0] line_stride;

assign line_stride=matrix>>(stride-1);

always @(posedge clk)
begin
if (conv_en==1)
	begin
		if (zagryzka_weight==0)
		begin
		   next_number = matrix;
		   next_number_prov = matrix;
		   if ((step!=3)&&(step!=12)&&(step!=24)&&(step!=36)&&(step!=72)) stride_plus=0;
		   else stride_plus=matrix;
		   if ((step!=3)&&(step!=12)&&(step!=24)&&(step!=36)&&(step!=72)) stride_plus_prov=0;
		   else stride_plus_prov=matrix;
		   case (marker)
				0: begin
				        re_wb=1;
				        read_addressw=memstartw+0*((depthwise)?1:((onexone)?((mem+1)>>3):(filt+1)));
				        read_addressb=memstartb+0;
				end
				1: begin
				        read_addressw=memstartw+1*((depthwise)?1:((onexone)?((mem+1)>>3):(filt+1)));
				        read_addressb=memstartb+1;
				end
				2: begin
				        read_addressw=memstartw+2*((depthwise)?1:((onexone)?((mem+1)>>3):(filt+1)));
				        read_addressb=memstartb+2;
				        w11_pre=qw[SIZE_weights-1:0]; 
				        w12_pre=qw[SIZE_weights*2-1:SIZE_weights]; 
				        w13_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; 
				        w14_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; 
				        w15_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; 
				        w16_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; 
				        w17_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; 
				        w18_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; 
				        w19_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; 

				        data_bias_1 = q_bias;
				end
				3: begin
				        read_addressw=memstartw+3*((depthwise)?1:((onexone)?((mem+1)>>3):(filt+1)));
				        read_addressb=memstartb+3;
				        w21_pre=qw[SIZE_weights-1:0]; 
				        w22_pre=qw[SIZE_weights*2-1:SIZE_weights]; 
				        w23_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; 
				        w24_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; 
				        w25_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; 
				        w26_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; 
				        w27_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; 
				        w28_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; 
				        w29_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; 

				        data_bias_2 = q_bias;
				end
				4: begin
				        read_addressw=memstartw+4*((depthwise)?1:((onexone)?((mem+1)>>3):(filt+1)));
				        read_addressb=memstartb+4;
				        w31_pre=qw[SIZE_weights-1:0]; 
				        w32_pre=qw[SIZE_weights*2-1:SIZE_weights]; 
				        w33_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; 
				        w34_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; 
				        w35_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; 
				        w36_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; 
				        w37_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; 
				        w38_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; 
				        w39_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; 

				        data_bias_3 = q_bias;
				end
				5: begin
				        read_addressw=memstartw+5*((depthwise)?1:((onexone)?((mem+1)>>3):(filt+1)));
				        read_addressb=memstartb+5;
				        w41_pre=qw[SIZE_weights-1:0]; 
				        w42_pre=qw[SIZE_weights*2-1:SIZE_weights]; 
				        w43_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; 
				        w44_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; 
				        w45_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; 
				        w46_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; 
				        w47_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; 
				        w48_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; 
				        w49_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; 

				        data_bias_4 = q_bias;
				end
				6: begin
				        read_addressw=memstartw+6*((depthwise)?1:((onexone)?((mem+1)>>3):(filt+1)));
				        read_addressb=memstartb+6;
				        w51_pre=qw[SIZE_weights-1:0]; 
				        w52_pre=qw[SIZE_weights*2-1:SIZE_weights]; 
				        w53_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; 
				        w54_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; 
				        w55_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; 
				        w56_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; 
				        w57_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; 
				        w58_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; 
				        w59_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; 

				        data_bias_5 = q_bias;
				end
				7: begin
				        read_addressw=memstartw+7*((depthwise)?1:((onexone)?((mem+1)>>3):(filt+1)));
				        read_addressb=memstartb+7;
				        w61_pre=qw[SIZE_weights-1:0]; 
				        w62_pre=qw[SIZE_weights*2-1:SIZE_weights]; 
				        w63_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; 
				        w64_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; 
				        w65_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; 
				        w66_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; 
				        w67_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; 
				        w68_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; 
				        w69_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; 

				        data_bias_6 = q_bias;
				end
				8: begin
				        w71_pre=qw[SIZE_weights-1:0]; 
				        w72_pre=qw[SIZE_weights*2-1:SIZE_weights]; 
				        w73_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; 
				        w74_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; 
				        w75_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; 
				        w76_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; 
				        w77_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; 
				        w78_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; 
				        w79_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; 

				        data_bias_7 = q_bias;
				end
				9: begin
				        w81_pre=qw[SIZE_weights-1:0]; 
				        w82_pre=qw[SIZE_weights*2-1:SIZE_weights]; 
				        w83_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; 
				        w84_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; 
				        w85_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; 
				        w86_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; 
				        w87_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; 
				        w88_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; 
				        w89_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; 

				        data_bias_8 = q_bias;
				        zagryzka_weight=1; re_wb=0; marker=-1;
				end
				default: 
					begin
						read_addressw=0;
						read_addressb=0;
						re_wb=0;
						$display("Check zagryzka_weight");
					end
		endcase
		marker=marker+1;
		end
		else
		begin
			re=1;
			case (marker)
				0: begin	
								re_t=0;
								if ((stride==2)&&(i==next_number))
									begin
										stride_plus=stride_plus+matrix;
										next_number = matrix+next_number;
									end
								if (onexone) read_addressp = memstartp+(matrix*matrix)*(3*i_onexone_1+marker)+i_onexone-1;
								else read_addressp=i+memstartp+stride_plus;

								if (onexone)
									begin
										p0_1=p6_pre;
										p0_2=p7_pre;
										p0_3=0;
										p1_1=p6_pre;
										p1_2=p7_pre;
										p1_3=0;
										p2_1=p6_pre;
										p2_2=p7_pre;
										p2_3=0;
										p3_1=p6_pre;
										p3_2=p7_pre;
										p3_3=0;
										p4_1=p6_pre;
										p4_2=p7_pre;
										p4_3=0;
										p5_1=p6_pre;
										p5_2=p7_pre;
										p5_3=0;
										p6_1=p6_pre;
										p6_2=p7_pre;
										p6_3=0;
										p7_1=p6_pre;
										p7_2=p7_pre;
										p7_3=0;
									end
								else
									begin
										if (depthwise)
											begin
												buff0_2[2]=qp[SIZE_8-1:SIZE_7];
												buff1_2[2]=qp[SIZE_7-1:SIZE_6];
												buff2_2[2]=qp[SIZE_6-1:SIZE_5];
												buff3_2[2]=qp[SIZE_5-1:SIZE_4];
												buff4_2[2]=qp[SIZE_4-1:SIZE_3];
												buff5_2[2]=qp[SIZE_3-1:SIZE_2];
												buff6_2[2]=qp[SIZE_2-1:SIZE_1];
												buff7_2[2]=qp[SIZE_1-1:0];
											end
										else
											begin
												if (((i+stride_plus-1)<matrix2-matrix)||(onexone))
													begin
													    if ({lvl[2],lvl[1],lvl[0]}==3'd0) 
															begin
																buff0_2[2]=qp[SIZE_8-1:SIZE_7];
																buff1_2[2]=qp[SIZE_8-1:SIZE_7];
																buff2_2[2]=qp[SIZE_8-1:SIZE_7];
																buff3_2[2]=qp[SIZE_8-1:SIZE_7];
																buff4_2[2]=qp[SIZE_8-1:SIZE_7];
																buff5_2[2]=qp[SIZE_8-1:SIZE_7];
																buff6_2[2]=qp[SIZE_8-1:SIZE_7];
																buff7_2[2]=qp[SIZE_8-1:SIZE_7];
															end
													    else if ({lvl[2],lvl[1],lvl[0]}==3'd1) 
															begin
																buff0_2[2]=qp[SIZE_7-1:SIZE_6];
																buff1_2[2]=qp[SIZE_7-1:SIZE_6];
																buff2_2[2]=qp[SIZE_7-1:SIZE_6];
																buff3_2[2]=qp[SIZE_7-1:SIZE_6];
																buff4_2[2]=qp[SIZE_7-1:SIZE_6];
																buff5_2[2]=qp[SIZE_7-1:SIZE_6];
																buff6_2[2]=qp[SIZE_7-1:SIZE_6];
																buff7_2[2]=qp[SIZE_7-1:SIZE_6];
															end
													    else if ({lvl[2],lvl[1],lvl[0]}==3'd2) 
															begin
																buff0_2[2]=qp[SIZE_6-1:SIZE_5];
																buff1_2[2]=qp[SIZE_6-1:SIZE_5];
																buff2_2[2]=qp[SIZE_6-1:SIZE_5];
																buff3_2[2]=qp[SIZE_6-1:SIZE_5];
																buff4_2[2]=qp[SIZE_6-1:SIZE_5];
																buff5_2[2]=qp[SIZE_6-1:SIZE_5];
																buff6_2[2]=qp[SIZE_6-1:SIZE_5];
																buff7_2[2]=qp[SIZE_6-1:SIZE_5];
															end
													    else if ({lvl[2],lvl[1],lvl[0]}==3'd3) 
															begin
																buff0_2[2]=qp[SIZE_5-1:SIZE_4];
																buff1_2[2]=qp[SIZE_5-1:SIZE_4];
																buff2_2[2]=qp[SIZE_5-1:SIZE_4];
																buff3_2[2]=qp[SIZE_5-1:SIZE_4];
																buff4_2[2]=qp[SIZE_5-1:SIZE_4];
																buff5_2[2]=qp[SIZE_5-1:SIZE_4];
																buff6_2[2]=qp[SIZE_5-1:SIZE_4];
																buff7_2[2]=qp[SIZE_5-1:SIZE_4];
															end
													    else if ({lvl[2],lvl[1],lvl[0]}==3'd4) 
															begin
																buff0_2[2]=qp[SIZE_4-1:SIZE_3];
																buff1_2[2]=qp[SIZE_4-1:SIZE_3];
																buff2_2[2]=qp[SIZE_4-1:SIZE_3];
																buff3_2[2]=qp[SIZE_4-1:SIZE_3];
																buff4_2[2]=qp[SIZE_4-1:SIZE_3];
																buff5_2[2]=qp[SIZE_4-1:SIZE_3];
																buff6_2[2]=qp[SIZE_4-1:SIZE_3];
																buff7_2[2]=qp[SIZE_4-1:SIZE_3];
															end
													    else if ({lvl[2],lvl[1],lvl[0]}==3'd5) 
															begin
																buff0_2[2]=qp[SIZE_3-1:SIZE_2];
																buff1_2[2]=qp[SIZE_3-1:SIZE_2];
																buff2_2[2]=qp[SIZE_3-1:SIZE_2];
																buff3_2[2]=qp[SIZE_3-1:SIZE_2];
																buff4_2[2]=qp[SIZE_3-1:SIZE_2];
																buff5_2[2]=qp[SIZE_3-1:SIZE_2];
																buff6_2[2]=qp[SIZE_3-1:SIZE_2];
																buff7_2[2]=qp[SIZE_3-1:SIZE_2];
															end
													    else if ({lvl[2],lvl[1],lvl[0]}==3'd6) 
															begin
																buff0_2[2]=qp[SIZE_2-1:SIZE_1];
																buff1_2[2]=qp[SIZE_2-1:SIZE_1];
																buff2_2[2]=qp[SIZE_2-1:SIZE_1];
																buff3_2[2]=qp[SIZE_2-1:SIZE_1];
																buff4_2[2]=qp[SIZE_2-1:SIZE_1];
																buff5_2[2]=qp[SIZE_2-1:SIZE_1];
																buff6_2[2]=qp[SIZE_2-1:SIZE_1];
																buff7_2[2]=qp[SIZE_2-1:SIZE_1];
															end
													    else if ({lvl[2],lvl[1],lvl[0]}==3'd7) 
															begin
																buff0_2[2]=qp[SIZE_1-1:0];
																buff1_2[2]=qp[SIZE_1-1:0];
																buff2_2[2]=qp[SIZE_1-1:0];
																buff3_2[2]=qp[SIZE_1-1:0];
																buff4_2[2]=qp[SIZE_1-1:0];
																buff5_2[2]=qp[SIZE_1-1:0];
																buff6_2[2]=qp[SIZE_1-1:0];
																buff7_2[2]=qp[SIZE_1-1:0];
															end
													end
												else
													begin
														buff0_2[2]=0;
														buff1_2[2]=0;
														buff2_2[2]=0;
														buff3_2[2]=0;
														buff4_2[2]=0;
														buff5_2[2]=0;
														buff6_2[2]=0;
														buff7_2[2]=0;
													end
											end
											p0_1=buff0_2[0];
											p0_2=buff0_2[1];
											p0_3=buff0_2[2];

											p1_1=buff1_2[0];
											p1_2=buff1_2[1];
											p1_3=buff1_2[2];

											p2_1=buff2_2[0];
											p2_2=buff2_2[1];
											p2_3=buff2_2[2];

											p3_1=buff3_2[0];
											p3_2=buff3_2[1];
											p3_3=buff3_2[2];

											p4_1=buff4_2[0];
											p4_2=buff4_2[1];
											p4_3=buff4_2[2];

											p5_1=buff5_2[0];
											p5_2=buff5_2[1];
											p5_3=buff5_2[2];

											p6_1=buff6_2[0];
											p6_2=buff6_2[1];
											p6_3=buff6_2[2];

											p7_1=buff7_2[0];
											p7_2=buff7_2[1];
											p7_3=buff7_2[2];

									end

								w11=(onexone)?w13_pre:w13_pre;
								w12=(onexone)?w12_pre:w12_pre;
								w13=(onexone)?w11_pre:w11_pre;
								w21=(onexone)?w23_pre:w23_pre;
								w22=(onexone)?w22_pre:w22_pre;
								w23=(onexone)?w21_pre:w21_pre;
								w31=(onexone)?w33_pre:w33_pre;
								w32=(onexone)?w32_pre:w32_pre;
								w33=(onexone)?w31_pre:w31_pre;
								w41=(onexone)?w43_pre:w43_pre;
								w42=(onexone)?w42_pre:w42_pre;
								w43=(onexone)?w41_pre:w41_pre;
								w51=(onexone)?w53_pre:w53_pre;
								w52=(onexone)?w52_pre:w52_pre;
								w53=(onexone)?w51_pre:w51_pre;
								w61=(onexone)?w63_pre:w63_pre;
								w62=(onexone)?w62_pre:w62_pre;
								w63=(onexone)?w61_pre:w61_pre;
								w71=(onexone)?w73_pre:w73_pre;
								w72=(onexone)?w72_pre:w72_pre;
								w73=(onexone)?w71_pre:w71_pre;
								w81=(onexone)?w83_pre:w83_pre;
								w82=(onexone)?w82_pre:w82_pre;
								w83=(onexone)?w81_pre:w81_pre;
								up_perm=0;
								if (onexone) down_perm=0; else down_perm=1;
								res1=Y1;
								res2=Y2;
								res3=Y3;
								res4=Y4;
								res5=Y5;
								res6=Y6;
								res7=Y7;
								res8=Y8;
					end
				1: begin
								if (onexone) read_addressp = memstartp+(matrix*matrix)*(3*i_onexone_1+marker)+i_onexone-1;
								else	if ((i+stride_plus)>=matrix-1)	read_addressp=i-matrix+memstartp+stride_plus;

								res1=res1+Y1;
								res2=res2+Y2;
								res3=res3+Y3;
								res4=res4+Y4;
								res5=res5+Y5;
								res6=res6+Y6;
								res7=res7+Y7;
								res8=res8+Y8;
								if ((i>=2)&&(((stride==2)&&((((step==3)||(step==12)||(step==24)||(step==36)||(step==72))&&(i[0]==1))||(((step!=3)&&(step!=12)&&(step!=24)&&(step!=36)&&(step!=72))&&(i[0]==0))))||(stride==1))) 
									begin
										res_old_1=qtp[32*8-1:32*7];
										res_old_2=qtp[32*7-1:32*6];
										res_old_3=qtp[32*6-1:32*5];
										res_old_4=qtp[32*5-1:32*4];
										res_old_5=qtp[32*4-1:32*3];
										res_old_6=qtp[32*3-1:32*2];
										res_old_7=qtp[32*2-1:32*1];
										res_old_8=qtp[32*1-1:32*0];
									end
								go=0;
								i_to_prov=i_to_prov+1'b1;
								if ((stride==2)&&(i_to_prov==next_number_prov)) 
									begin
										stride_plus_prov=stride_plus_prov+matrix;
										next_number_prov = matrix+next_number_prov;
									end

								buff0_2[0]=buff0_2[1];
								buff0_1[0]=buff0_1[1];
								buff0_0[0]=buff0_0[1];
								buff0_2[1]=buff0_2[2];
								buff0_1[1]=buff0_1[2];
								buff0_0[1]=buff0_0[2];

								buff1_2[0]=buff1_2[1];
								buff1_1[0]=buff1_1[1];
								buff1_0[0]=buff1_0[1];
								buff1_2[1]=buff1_2[2];
								buff1_1[1]=buff1_1[2];
								buff1_0[1]=buff1_0[2];

								buff2_2[0]=buff2_2[1];
								buff2_1[0]=buff2_1[1];
								buff2_0[0]=buff2_0[1];
								buff2_2[1]=buff2_2[2];
								buff2_1[1]=buff2_1[2];
								buff2_0[1]=buff2_0[2];

								buff3_2[0]=buff3_2[1];
								buff3_1[0]=buff3_1[1];
								buff3_0[0]=buff3_0[1];
								buff3_2[1]=buff3_2[2];
								buff3_1[1]=buff3_1[2];
								buff3_0[1]=buff3_0[2];

								buff4_2[0]=buff4_2[1];
								buff4_1[0]=buff4_1[1];
								buff4_0[0]=buff4_0[1];
								buff4_2[1]=buff4_2[2];
								buff4_1[1]=buff4_1[2];
								buff4_0[1]=buff4_0[2];

								buff5_2[0]=buff5_2[1];
								buff5_1[0]=buff5_1[1];
								buff5_0[0]=buff5_0[1];
								buff5_2[1]=buff5_2[2];
								buff5_1[1]=buff5_1[2];
								buff5_0[1]=buff5_0[2];

								buff6_2[0]=buff6_2[1];
								buff6_1[0]=buff6_1[1];
								buff6_0[0]=buff6_0[1];
								buff6_2[1]=buff6_2[2];
								buff6_1[1]=buff6_1[2];
								buff6_0[1]=buff6_0[2];

								buff7_2[0]=buff7_2[1];
								buff7_1[0]=buff7_1[1];
								buff7_0[0]=buff7_0[1];
								buff7_2[1]=buff7_2[2];
								buff7_1[1]=buff7_1[2];
								buff7_0[1]=buff7_0[2];

					end
				2: begin
							if (onexone) read_addressp = memstartp+(matrix*matrix)*(3*i_onexone_1+marker)+i_onexone-1;
							else	if ((i+stride_plus)<matrix2-matrix) read_addressp=i+matrix+memstartp+stride_plus;

							if (onexone)
								begin
									p0_pre = qp[SIZE_8-1:SIZE_7];
									p1_pre = qp[SIZE_7-1:SIZE_6];
									p2_pre = qp[SIZE_6-1:SIZE_5];
									p3_pre = qp[SIZE_5-1:SIZE_4];
									p4_pre = qp[SIZE_4-1:SIZE_3];
									p5_pre = qp[SIZE_3-1:SIZE_2];
									p6_pre = qp[SIZE_2-1:SIZE_1];
									p7_pre = qp[SIZE_1-1:0];
									p0_1=p0_pre;
									p0_2=p1_pre;
									p0_3=p2_pre;

									p1_1=p0_pre;
									p1_2=p1_pre;
									p1_3=p2_pre;

									p2_1=p0_pre;
									p2_2=p1_pre;
									p2_3=p2_pre;

									p3_1=p0_pre;
									p3_2=p1_pre;
									p3_3=p2_pre;

									p4_1=p0_pre;
									p4_2=p1_pre;
									p4_3=p2_pre;

									p5_1=p0_pre;
									p5_2=p1_pre;
									p5_3=p2_pre;

									p6_1=p0_pre;
									p6_2=p1_pre;
									p6_3=p2_pre;

									p7_1=p0_pre;
									p7_2=p1_pre;
									p7_3=p2_pre;

								end
							else
								begin
									if (depthwise)
										begin
											buff0_1[2]=qp[SIZE_8-1:SIZE_7];
											buff1_1[2]=qp[SIZE_7-1:SIZE_6];
											buff2_1[2]=qp[SIZE_6-1:SIZE_5];
											buff3_1[2]=qp[SIZE_5-1:SIZE_4];
											buff4_1[2]=qp[SIZE_4-1:SIZE_3];
											buff5_1[2]=qp[SIZE_3-1:SIZE_2];
											buff6_1[2]=qp[SIZE_2-1:SIZE_1];
											buff7_1[2]=qp[SIZE_1-1:0];
										end
									else
										begin
											if ({lvl[2],lvl[1],lvl[0]}==3'd0) 
												begin
													buff0_1[2]=qp[SIZE_8-1:SIZE_7];
													buff1_1[2]=qp[SIZE_8-1:SIZE_7];
													buff2_1[2]=qp[SIZE_8-1:SIZE_7];
													buff3_1[2]=qp[SIZE_8-1:SIZE_7];
													buff4_1[2]=qp[SIZE_8-1:SIZE_7];
													buff5_1[2]=qp[SIZE_8-1:SIZE_7];
													buff6_1[2]=qp[SIZE_8-1:SIZE_7];
													buff7_1[2]=qp[SIZE_8-1:SIZE_7];
												end
											else if ({lvl[2],lvl[1],lvl[0]}==3'd1) 
												begin
													buff0_1[2]=qp[SIZE_7-1:SIZE_6];
													buff1_1[2]=qp[SIZE_7-1:SIZE_6];
													buff2_1[2]=qp[SIZE_7-1:SIZE_6];
													buff3_1[2]=qp[SIZE_7-1:SIZE_6];
													buff4_1[2]=qp[SIZE_7-1:SIZE_6];
													buff5_1[2]=qp[SIZE_7-1:SIZE_6];
													buff6_1[2]=qp[SIZE_7-1:SIZE_6];
													buff7_1[2]=qp[SIZE_7-1:SIZE_6];
												end
											else if ({lvl[2],lvl[1],lvl[0]}==3'd2) 
												begin
													buff0_1[2]=qp[SIZE_6-1:SIZE_5];
													buff1_1[2]=qp[SIZE_6-1:SIZE_5];
													buff2_1[2]=qp[SIZE_6-1:SIZE_5];
													buff3_1[2]=qp[SIZE_6-1:SIZE_5];
													buff4_1[2]=qp[SIZE_6-1:SIZE_5];
													buff5_1[2]=qp[SIZE_6-1:SIZE_5];
													buff6_1[2]=qp[SIZE_6-1:SIZE_5];
													buff7_1[2]=qp[SIZE_6-1:SIZE_5];
												end
											else if ({lvl[2],lvl[1],lvl[0]}==3'd3) 
												begin
													buff0_1[2]=qp[SIZE_5-1:SIZE_4];
													buff1_1[2]=qp[SIZE_5-1:SIZE_4];
													buff2_1[2]=qp[SIZE_5-1:SIZE_4];
													buff3_1[2]=qp[SIZE_5-1:SIZE_4];
													buff4_1[2]=qp[SIZE_5-1:SIZE_4];
													buff5_1[2]=qp[SIZE_5-1:SIZE_4];
													buff6_1[2]=qp[SIZE_5-1:SIZE_4];
													buff7_1[2]=qp[SIZE_5-1:SIZE_4];
												end
											else if ({lvl[2],lvl[1],lvl[0]}==3'd4) 
												begin
													buff0_1[2]=qp[SIZE_4-1:SIZE_3];
													buff1_1[2]=qp[SIZE_4-1:SIZE_3];
													buff2_1[2]=qp[SIZE_4-1:SIZE_3];
													buff3_1[2]=qp[SIZE_4-1:SIZE_3];
													buff4_1[2]=qp[SIZE_4-1:SIZE_3];
													buff5_1[2]=qp[SIZE_4-1:SIZE_3];
													buff6_1[2]=qp[SIZE_4-1:SIZE_3];
													buff7_1[2]=qp[SIZE_4-1:SIZE_3];
												end
											else if ({lvl[2],lvl[1],lvl[0]}==3'd5) 
												begin
													buff0_1[2]=qp[SIZE_3-1:SIZE_2];
													buff1_1[2]=qp[SIZE_3-1:SIZE_2];
													buff2_1[2]=qp[SIZE_3-1:SIZE_2];
													buff3_1[2]=qp[SIZE_3-1:SIZE_2];
													buff4_1[2]=qp[SIZE_3-1:SIZE_2];
													buff5_1[2]=qp[SIZE_3-1:SIZE_2];
													buff6_1[2]=qp[SIZE_3-1:SIZE_2];
													buff7_1[2]=qp[SIZE_3-1:SIZE_2];
												end
											else if ({lvl[2],lvl[1],lvl[0]}==3'd6) 
												begin
													buff0_1[2]=qp[SIZE_2-1:SIZE_1];
													buff1_1[2]=qp[SIZE_2-1:SIZE_1];
													buff2_1[2]=qp[SIZE_2-1:SIZE_1];
													buff3_1[2]=qp[SIZE_2-1:SIZE_1];
													buff4_1[2]=qp[SIZE_2-1:SIZE_1];
													buff5_1[2]=qp[SIZE_2-1:SIZE_1];
													buff6_1[2]=qp[SIZE_2-1:SIZE_1];
													buff7_1[2]=qp[SIZE_2-1:SIZE_1];
												end
											else if ({lvl[2],lvl[1],lvl[0]}==3'd7) 
												begin
													buff0_1[2]=qp[SIZE_1-1:0];
													buff1_1[2]=qp[SIZE_1-1:0];
													buff2_1[2]=qp[SIZE_1-1:0];
													buff3_1[2]=qp[SIZE_1-1:0];
													buff4_1[2]=qp[SIZE_1-1:0];
													buff5_1[2]=qp[SIZE_1-1:0];
													buff6_1[2]=qp[SIZE_1-1:0];
													buff7_1[2]=qp[SIZE_1-1:0];
												end
										end
										p0_1=buff0_1[0];
										p0_2=buff0_1[1];
										p0_3=buff0_1[2];

										p1_1=buff1_1[0];
										p1_2=buff1_1[1];
										p1_3=buff1_1[2];

										p2_1=buff2_1[0];
										p2_2=buff2_1[1];
										p2_3=buff2_1[2];

										p3_1=buff3_1[0];
										p3_2=buff3_1[1];
										p3_3=buff3_1[2];

										p4_1=buff4_1[0];
										p4_2=buff4_1[1];
										p4_3=buff4_1[2];

										p5_1=buff5_1[0];
										p5_2=buff5_1[1];
										p5_3=buff5_1[2];

										p6_1=buff6_1[0];
										p6_2=buff6_1[1];
										p6_3=buff6_1[2];

										p7_1=buff7_1[0];
										p7_2=buff7_1[1];
										p7_3=buff7_1[2];

								end
								w11=(onexone)?w19_pre:w16_pre;
								w12=(onexone)?w18_pre:w15_pre;
								w13=(onexone)?w17_pre:w14_pre;

								w21=(onexone)?w29_pre:w26_pre;
								w22=(onexone)?w28_pre:w25_pre;
								w23=(onexone)?w27_pre:w24_pre;

								w31=(onexone)?w39_pre:w36_pre;
								w32=(onexone)?w38_pre:w35_pre;
								w33=(onexone)?w37_pre:w34_pre;

								w41=(onexone)?w49_pre:w46_pre;
								w42=(onexone)?w48_pre:w45_pre;
								w43=(onexone)?w47_pre:w44_pre;

								w51=(onexone)?w59_pre:w56_pre;
								w52=(onexone)?w58_pre:w55_pre;
								w53=(onexone)?w57_pre:w54_pre;

								w61=(onexone)?w69_pre:w66_pre;
								w62=(onexone)?w68_pre:w65_pre;
								w63=(onexone)?w67_pre:w64_pre;

								w71=(onexone)?w79_pre:w76_pre;
								w72=(onexone)?w78_pre:w75_pre;
								w73=(onexone)?w77_pre:w74_pre;

								w81=(onexone)?w89_pre:w86_pre;
								w82=(onexone)?w88_pre:w85_pre;
								w83=(onexone)?w87_pre:w84_pre;

								go=1;
								up_perm=0;
								down_perm=0;
								if ((i>=2)&&(((stride==2)&&((((step==3)||(step==12)||(step==24)||(step==36)||(step==72))&&(i[0]==1))||(((step!=3)&&(step!=12)&&(step!=24)&&(step!=36)&&(step!=72))&&(i[0]==0))))||(stride==1)))
								begin
								if (onexone) write_addresstp=i_onexone-2;
								else write_addresstp=(i>>(stride-1))-1;
								if (glob_average_en)  write_addressp=memstartzap;
								else
									begin
										if (onexone)	write_addressp=memstartzap+i_onexone-2;
										else			write_addressp=memstartzap+((i-2)>>(stride-1));
									end

								if (((onexone && (i_onexone_1 == 0)) || !onexone)&&(!bias)) we_t=1;

								res1=res1+Y1;
								res2=res2+Y2;
								res3=res3+Y3;
								res4=res4+Y4;
								res5=res5+Y5;
								res6=res6+Y6;
								res7=res7+Y7;
								res8=res8+Y8;

								if ((lvl!=0)&&(!depthwise))
									begin
										res1=res1+res_old_1;
										res2=res2+res_old_2;
										res3=res3+res_old_3;
										res4=res4+res_old_4;
										res5=res5+res_old_5;
										res6=res6+res_old_6;
										res7=res7+res_old_7;
										res8=res8+res_old_8;
									end
								if (bias)
									begin
										res1=res1+(data_bias_1<<13);
										res2=res2+(data_bias_2<<13);
										res3=res3+(data_bias_3<<13);
										res4=res4+(data_bias_4<<13);
										res5=res5+(data_bias_5<<13);
										res6=res6+(data_bias_6<<13);
										res7=res7+(data_bias_7<<13);
										res8=res8+(data_bias_8<<13);

										if (res1<0) res1=0;  //RELU
										if (res2<0) res2=0;  //RELU
										if (res3<0) res3=0;  //RELU
										if (res4<0) res4=0;  //RELU
										if (res5<0) res5=0;  //RELU
										if (res6<0) res6=0;  //RELU
										if (res7<0) res7=0;  //RELU
										if (res8<0) res8=0;  //RELU

										res_bias_check_1=res1[32-1-2:SIZE_1-2];
										res_bias_check_2=res2[32-1-2:SIZE_1-2];
										res_bias_check_3=res3[32-1-2:SIZE_1-2];
										res_bias_check_4=res4[32-1-2:SIZE_1-2];
										res_bias_check_5=res5[32-1-2:SIZE_1-2];
										res_bias_check_6=res6[32-1-2:SIZE_1-2];
										res_bias_check_7=res7[32-1-2:SIZE_1-2];
										res_bias_check_8=res8[32-1-2:SIZE_1-2];

										if (res_bias_check_1>(2**(SIZE_1-1))-1) res_out_1=(2**(SIZE_1-1))-1;
										else res_out_1=res1[SIZE_1+SIZE_1-2-2:SIZE_1-2];
										if (res_bias_check_2>(2**(SIZE_1-1))-1) res_out_2=(2**(SIZE_1-1))-1;
										else res_out_2=res2[SIZE_1+SIZE_1-2-2:SIZE_1-2];
										if (res_bias_check_3>(2**(SIZE_1-1))-1) res_out_3=(2**(SIZE_1-1))-1;
										else res_out_3=res3[SIZE_1+SIZE_1-2-2:SIZE_1-2];
										if (res_bias_check_4>(2**(SIZE_1-1))-1) res_out_4=(2**(SIZE_1-1))-1;
										else res_out_4=res4[SIZE_1+SIZE_1-2-2:SIZE_1-2];
										if (res_bias_check_5>(2**(SIZE_1-1))-1) res_out_5=(2**(SIZE_1-1))-1;
										else res_out_5=res5[SIZE_1+SIZE_1-2-2:SIZE_1-2];
										if (res_bias_check_6>(2**(SIZE_1-1))-1) res_out_6=(2**(SIZE_1-1))-1;
										else res_out_6=res6[SIZE_1+SIZE_1-2-2:SIZE_1-2];
										if (res_bias_check_7>(2**(SIZE_1-1))-1) res_out_7=(2**(SIZE_1-1))-1;
										else res_out_7=res7[SIZE_1+SIZE_1-2-2:SIZE_1-2];
										if (res_bias_check_8>(2**(SIZE_1-1))-1) res_out_8=(2**(SIZE_1-1))-1;
										else res_out_8=res8[SIZE_1+SIZE_1-2-2:SIZE_1-2];

										if ((glob_average_en)&&(i_onexone_1 == 0))
											begin
												glob_average_perem_1 = glob_average_perem_1 + res_out_1;
												glob_average_perem_2 = glob_average_perem_2 + res_out_2;
												glob_average_perem_3 = glob_average_perem_3 + res_out_3;
												glob_average_perem_4 = glob_average_perem_4 + res_out_4;
												glob_average_perem_5 = glob_average_perem_5 + res_out_5;
												glob_average_perem_6 = glob_average_perem_6 + res_out_6;
												glob_average_perem_7 = glob_average_perem_7 + res_out_7;
												glob_average_perem_8 = glob_average_perem_8 + res_out_8;
											end
										if ((onexone && (i_onexone_1 == 0)) || !onexone) we=1;
									end
								end
					end
				3: begin
								re_t=1;
								if (onexone) read_addresstp=i_onexone-1;
								else read_addresstp=(i>>(stride-1))-1;

								if (onexone)
									begin
										p8_pre = qp[SIZE_8-1:SIZE_7];
										p9_pre = qp[SIZE_7-1:SIZE_6];
										p10_pre = qp[SIZE_6-1:SIZE_5];
										p11_pre = qp[SIZE_5-1:SIZE_4];
										p12_pre = qp[SIZE_4-1:SIZE_3];
										p13_pre = qp[SIZE_3-1:SIZE_2];
										p14_pre = qp[SIZE_2-1:SIZE_1];
										p15_pre = qp[SIZE_1-1:0];

										p0_1=p3_pre;
										p0_2=p4_pre;
										p0_3=p5_pre;

										p1_1=p3_pre;
										p1_2=p4_pre;
										p1_3=p5_pre;

										p2_1=p3_pre;
										p2_2=p4_pre;
										p2_3=p5_pre;

										p3_1=p3_pre;
										p3_2=p4_pre;
										p3_3=p5_pre;

										p4_1=p3_pre;
										p4_2=p4_pre;
										p4_3=p5_pre;

										p5_1=p3_pre;
										p5_2=p4_pre;
										p5_3=p5_pre;

										p6_1=p3_pre;
										p6_2=p4_pre;
										p6_3=p5_pre;

										p7_1=p3_pre;
										p7_2=p4_pre;
										p7_3=p5_pre;

									end
								else
									begin
										if (depthwise)
											begin
												buff0_0[2]=qp[SIZE_8-1:SIZE_7];
												buff1_0[2]=qp[SIZE_7-1:SIZE_6];
												buff2_0[2]=qp[SIZE_6-1:SIZE_5];
												buff3_0[2]=qp[SIZE_5-1:SIZE_4];
												buff4_0[2]=qp[SIZE_4-1:SIZE_3];
												buff5_0[2]=qp[SIZE_3-1:SIZE_2];
												buff6_0[2]=qp[SIZE_2-1:SIZE_1];
												buff7_0[2]=qp[SIZE_1-1:0];
											end
										else
											begin
												if ((i+stride_plus)>=matrix-1)
												begin
													if ({lvl[2],lvl[1],lvl[0]}==3'd0) 
														begin
															buff0_0[2]=qp[SIZE_8-1:SIZE_7];
															buff1_0[2]=qp[SIZE_8-1:SIZE_7];
															buff2_0[2]=qp[SIZE_8-1:SIZE_7];
															buff3_0[2]=qp[SIZE_8-1:SIZE_7];
															buff4_0[2]=qp[SIZE_8-1:SIZE_7];
															buff5_0[2]=qp[SIZE_8-1:SIZE_7];
															buff6_0[2]=qp[SIZE_8-1:SIZE_7];
															buff7_0[2]=qp[SIZE_8-1:SIZE_7];
														end
													else if ({lvl[2],lvl[1],lvl[0]}==3'd1) 
														begin
															buff0_0[2]=qp[SIZE_7-1:SIZE_6];
															buff1_0[2]=qp[SIZE_7-1:SIZE_6];
															buff2_0[2]=qp[SIZE_7-1:SIZE_6];
															buff3_0[2]=qp[SIZE_7-1:SIZE_6];
															buff4_0[2]=qp[SIZE_7-1:SIZE_6];
															buff5_0[2]=qp[SIZE_7-1:SIZE_6];
															buff6_0[2]=qp[SIZE_7-1:SIZE_6];
															buff7_0[2]=qp[SIZE_7-1:SIZE_6];
														end
													else if ({lvl[2],lvl[1],lvl[0]}==3'd2) 
														begin
															buff0_0[2]=qp[SIZE_6-1:SIZE_5];
															buff1_0[2]=qp[SIZE_6-1:SIZE_5];
															buff2_0[2]=qp[SIZE_6-1:SIZE_5];
															buff3_0[2]=qp[SIZE_6-1:SIZE_5];
															buff4_0[2]=qp[SIZE_6-1:SIZE_5];
															buff5_0[2]=qp[SIZE_6-1:SIZE_5];
															buff6_0[2]=qp[SIZE_6-1:SIZE_5];
															buff7_0[2]=qp[SIZE_6-1:SIZE_5];
														end
													else if ({lvl[2],lvl[1],lvl[0]}==3'd3) 
														begin
															buff0_0[2]=qp[SIZE_5-1:SIZE_4];
															buff1_0[2]=qp[SIZE_5-1:SIZE_4];
															buff2_0[2]=qp[SIZE_5-1:SIZE_4];
															buff3_0[2]=qp[SIZE_5-1:SIZE_4];
															buff4_0[2]=qp[SIZE_5-1:SIZE_4];
															buff5_0[2]=qp[SIZE_5-1:SIZE_4];
															buff6_0[2]=qp[SIZE_5-1:SIZE_4];
															buff7_0[2]=qp[SIZE_5-1:SIZE_4];
														end
													else if ({lvl[2],lvl[1],lvl[0]}==3'd4) 
														begin
															buff0_0[2]=qp[SIZE_4-1:SIZE_3];
															buff1_0[2]=qp[SIZE_4-1:SIZE_3];
															buff2_0[2]=qp[SIZE_4-1:SIZE_3];
															buff3_0[2]=qp[SIZE_4-1:SIZE_3];
															buff4_0[2]=qp[SIZE_4-1:SIZE_3];
															buff5_0[2]=qp[SIZE_4-1:SIZE_3];
															buff6_0[2]=qp[SIZE_4-1:SIZE_3];
															buff7_0[2]=qp[SIZE_4-1:SIZE_3];
														end
													else if ({lvl[2],lvl[1],lvl[0]}==3'd5) 
														begin
															buff0_0[2]=qp[SIZE_3-1:SIZE_2];
															buff1_0[2]=qp[SIZE_3-1:SIZE_2];
															buff2_0[2]=qp[SIZE_3-1:SIZE_2];
															buff3_0[2]=qp[SIZE_3-1:SIZE_2];
															buff4_0[2]=qp[SIZE_3-1:SIZE_2];
															buff5_0[2]=qp[SIZE_3-1:SIZE_2];
															buff6_0[2]=qp[SIZE_3-1:SIZE_2];
															buff7_0[2]=qp[SIZE_3-1:SIZE_2];
														end
													else if ({lvl[2],lvl[1],lvl[0]}==3'd6) 
														begin
															buff0_0[2]=qp[SIZE_2-1:SIZE_1];
															buff1_0[2]=qp[SIZE_2-1:SIZE_1];
															buff2_0[2]=qp[SIZE_2-1:SIZE_1];
															buff3_0[2]=qp[SIZE_2-1:SIZE_1];
															buff4_0[2]=qp[SIZE_2-1:SIZE_1];
															buff5_0[2]=qp[SIZE_2-1:SIZE_1];
															buff6_0[2]=qp[SIZE_2-1:SIZE_1];
															buff7_0[2]=qp[SIZE_2-1:SIZE_1];
														end
													else if ({lvl[2],lvl[1],lvl[0]}==3'd7) 
														begin
															buff0_0[2]=qp[SIZE_1-1:0];
															buff1_0[2]=qp[SIZE_1-1:0];
															buff2_0[2]=qp[SIZE_1-1:0];
															buff3_0[2]=qp[SIZE_1-1:0];
															buff4_0[2]=qp[SIZE_1-1:0];
															buff5_0[2]=qp[SIZE_1-1:0];
															buff6_0[2]=qp[SIZE_1-1:0];
															buff7_0[2]=qp[SIZE_1-1:0];
														end
												end
												else
													begin
														buff0_0[2]=0;
														buff1_0[2]=0;
														buff2_0[2]=0;
														buff3_0[2]=0;
														buff4_0[2]=0;
														buff5_0[2]=0;
														buff6_0[2]=0;
														buff7_0[2]=0;
													end
											end
										p0_1=buff0_0[0];
										p0_2=buff0_0[1];
										p0_3=buff0_0[2];

										p1_1=buff1_0[0];
										p1_2=buff1_0[1];
										p1_3=buff1_0[2];

										p2_1=buff2_0[0];
										p2_2=buff2_0[1];
										p2_3=buff2_0[2];

										p3_1=buff3_0[0];
										p3_2=buff3_0[1];
										p3_3=buff3_0[2];

										p4_1=buff4_0[0];
										p4_2=buff4_0[1];
										p4_3=buff4_0[2];

										p5_1=buff5_0[0];
										p5_2=buff5_0[1];
										p5_3=buff5_0[2];

										p6_1=buff6_0[0];
										p6_2=buff6_0[1];
										p6_3=buff6_0[2];

										p7_1=buff7_0[0];
										p7_2=buff7_0[1];
										p7_3=buff7_0[2];

									end
								w11=(onexone)?w16_pre:w19_pre;
								w12=(onexone)?w15_pre:w18_pre;
								w13=(onexone)?w14_pre:w17_pre;

								w21=(onexone)?w26_pre:w29_pre;
								w22=(onexone)?w25_pre:w28_pre;
								w23=(onexone)?w24_pre:w27_pre;

								w31=(onexone)?w36_pre:w39_pre;
								w32=(onexone)?w35_pre:w38_pre;
								w33=(onexone)?w34_pre:w37_pre;

								w41=(onexone)?w46_pre:w49_pre;
								w42=(onexone)?w45_pre:w48_pre;
								w43=(onexone)?w44_pre:w47_pre;

								w51=(onexone)?w56_pre:w59_pre;
								w52=(onexone)?w55_pre:w58_pre;
								w53=(onexone)?w54_pre:w57_pre;

								w61=(onexone)?w66_pre:w69_pre;
								w62=(onexone)?w65_pre:w68_pre;
								w63=(onexone)?w64_pre:w67_pre;

								w71=(onexone)?w76_pre:w79_pre;
								w72=(onexone)?w75_pre:w78_pre;
								w73=(onexone)?w74_pre:w77_pre;

								w81=(onexone)?w86_pre:w89_pre;
								w82=(onexone)?w85_pre:w88_pre;
								w83=(onexone)?w84_pre:w87_pre;

								if (onexone) up_perm=0; else up_perm=1;
								down_perm=0;
								we_t=0;
								we=0;
					end		
			default: $display("Check case conv_TOP");
			endcase

			if (marker!=3) marker=marker+1;
			else begin 
					marker=0; 
					if (((i<matrix*line_stride+1)&&(!onexone))||((onexone)&&(i_onexone_plus1<(matrix*line_stride)+2)))
						begin
							i=i+1; 
							if (onexone)
								begin
									if (i_onexone_1 == 2>>2)
										begin
											i_onexone = i_onexone + 1;
											i_onexone_1 = 0;
										end
									else	i_onexone_1 = i_onexone_1 + 1;
								end
						end
					else STOP=1; 
				  end
		end
	end
else 
	begin
		i=0;
		i_to_prov=-2;
		stride_plus=0;
		next_number=matrix;
		zagryzka_weight=0;
		STOP=0;
		re=0;
		re_t=0;
		go=0;
		marker=0;
		glob_average_perem_1=0;
		glob_average_perem_2=0;
		glob_average_perem_3=0;
		glob_average_perem_4=0;
		glob_average_perem_5=0;
		glob_average_perem_6=0;
		glob_average_perem_7=0;
		glob_average_perem_8=0;
		i_onexone = 0;
		i_onexone_1 = 0;
		read_addressw=0;
		read_addressb=0;
		re_wb=0;
	end
end
assign glob_average_perem_1_1=glob_average_perem_1>>4;
assign glob_average_perem_2_1=glob_average_perem_2>>4;
assign glob_average_perem_3_1=glob_average_perem_3>>4;
assign glob_average_perem_4_1=glob_average_perem_4>>4;
assign glob_average_perem_5_1=glob_average_perem_5>>4;
assign glob_average_perem_6_1=glob_average_perem_6>>4;
assign glob_average_perem_7_1=glob_average_perem_7>>4;
assign glob_average_perem_8_1=glob_average_perem_8>>4;
assign dp={(glob_average_en?glob_average_perem_1_1:res_out_1),
(glob_average_en?glob_average_perem_2_1:res_out_2),
(glob_average_en?glob_average_perem_3_1:res_out_3),
(glob_average_en?glob_average_perem_4_1:res_out_4),
(glob_average_en?glob_average_perem_5_1:res_out_5),
(glob_average_en?glob_average_perem_6_1:res_out_6),
(glob_average_en?glob_average_perem_7_1:res_out_7),
(glob_average_en?glob_average_perem_8_1:res_out_8)
};
assign dtp={res1,res2,res3,res4,res5,res6,res7,res8};
endmodule
