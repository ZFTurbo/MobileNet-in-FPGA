module TOP(
clk,
clk_RAM_w,
clk_RAM_p,
GO,
RESULT,
STOP,

re_weights,
load_weights,
dp_weights,
address_weights,

re_bias,
load_bias,
dp_bias,
address_bias,

we_image,
dp_image,
address_image,
step
);

parameter num_conv=8;
parameter SIZE_weights = 19;
parameter SIZE_bias = 14;
parameter SIZE_1=13;
parameter SIZE_2=26;
parameter SIZE_3=39;
parameter SIZE_4=52;
parameter SIZE_5=65;
parameter SIZE_6=78;
parameter SIZE_7=91;
parameter SIZE_8=104;
parameter SIZE_address_pix=18;
parameter SIZE_address_pix_t=17;
parameter SIZE_address_wei=17;
parameter SIZE_address_image=16;
parameter picture_size = 128;
parameter picture_storage_limit = 0;
parameter razmpar = picture_size >> 1;
parameter razmpar2  = picture_size >> 2;
parameter picture_storage_limit_2 = picture_size*picture_size*1;
input clk,clk_RAM_w,clk_RAM_p;
input GO;
output [1:0] RESULT;
input signed [SIZE_weights-1:0] dp_weights;
input signed [SIZE_bias-1:0] dp_bias;
output [23:0] address_weights;
output [11:0] address_bias;
input load_weights,load_bias;
input signed [SIZE_1-1:0] dp_image;
input [SIZE_address_image-1:0] address_image;
input we_image;
output reg STOP;
output re_weights,re_bias;
output [6:0] step;

wire [SIZE_address_image-1:0] address_image_1;

reg conv_en;
wire STOP_conv;

reg dense_en;
wire STOP_dense;

reg result_en;
wire STOP_res;	
wire [1:0] res_out;

reg bias,glob_average_en;

reg [4:0] TOPlvl_conv;
wire [4:0] TOPlvl;
reg [8:0] lvl;
reg [8:0] slvl;
reg [2:0] num;
reg [SIZE_address_pix-1:0] memstartp;
wire [SIZE_address_pix-1:0] memstartp_lvl;
reg [SIZE_address_wei-1:0] memstartw;
wire [SIZE_address_wei-1:0] memstartw_lvl;
reg [SIZE_address_pix-1:0] memstartzap;
wire [SIZE_address_pix-1:0] memstartzap_num;
wire [10:0] 				memstartb;
wire [SIZE_address_pix-1:0] read_addressp;
wire [SIZE_address_image-1:0] read_addressp_init;
wire [SIZE_address_pix_t-1:0] read_addresstp;
wire [SIZE_address_wei-1:0] read_addressw;
wire [10:0]					read_address_bias; 
wire [SIZE_address_pix-1:0] read_addressp_conv;
wire [SIZE_address_pix-1:0] read_addressp_dense;
wire [SIZE_address_pix-1:0] read_addressp_res;
wire [SIZE_address_wei-1:0] read_addressw_conv;
wire [SIZE_address_wei-1:0] read_addressw_dense;
wire [SIZE_address_pix-1:0] write_addressp;
wire [SIZE_address_pix_t-1:0] write_addresstp;
wire [SIZE_address_wei-1:0] write_addressw;
wire [10:0]					write_address_bias; 
wire [SIZE_address_pix-1:0] write_addressp_zagr;
wire [SIZE_address_pix-1:0] write_addressp_conv;
wire [SIZE_address_pix-1:0] write_addressp_dense;
wire we_p,we_tp,we_w;
wire re_p,re_tp,re_w,re_p_init;
wire re_bias_RAM;
wire we_p_zagr;
wire we_conv,re_wb_conv,re_conv;
wire we_dense,re_p_dense,re_w_dense;
wire we_bias;
wire re_p_res;
wire signed [SIZE_8-1:0] qp;
wire signed [32*8-1:0] qtp;
wire signed [SIZE_weights*9-1:0] qw;
wire signed [SIZE_bias-1:0]	q_bias;
wire signed [SIZE_8-1:0] dp;
wire signed [32*8-1:0] dtp;
wire signed [SIZE_weights*9-1:0] dw;
wire signed [SIZE_8-1:0] dp_conv;
wire signed [SIZE_8-1:0] dp_dense;
wire signed [SIZE_8-1:0] dp_zagr;
wire signed [SIZE_bias-1:0] d_bias;

wire [1:0] prov;
wire [14:0] i_conv;
wire signed [32-1:0] Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8;

wire signed [SIZE_weights-1:0] w11,w12,w13,w21,w22,w23,w31,w32,w33,w41,w42,w43,w51,w52,w53,w61,w62,w63,w71,w72,w73,w81,w82,w83;
wire signed [SIZE_weights-1:0] w11_c,w12_c,w13_c,w21_c,w22_c,w23_c,w31_c,w32_c,w33_c,w41_c,w42_c,w43_c,w51_c,w52_c,w53_c,w61_c,w62_c,w63_c,w71_c,w72_c,w73_c,w81_c,w82_c,w83_c;
wire signed [SIZE_weights-1:0] w11_d,w12_d,w13_d,w21_d,w22_d,w23_d,w31_d,w32_d,w33_d,w41_d,w42_d,w43_d,w51_d,w52_d,w53_d,w61_d,w62_d,w63_d,w71_d,w72_d,w73_d,w81_d,w82_d,w83_d;
wire signed [SIZE_1-1:0] p11,p12,p13,p21,p22,p23,p31,p32,p33,p41,p42,p43,p51,p52,p53,p61,p62,p63,p71,p72,p73,p81,p82,p83;
wire signed [SIZE_1-1:0] p11_c,p12_c,p13_c,p21_c,p22_c,p23_c,p31_c,p32_c,p33_c,p41_c,p42_c,p43_c,p51_c,p52_c,p53_c,p61_c,p62_c,p63_c,p71_c,p72_c,p73_c,p81_c,p82_c,p83_c;
wire signed [SIZE_1-1:0] p11_d,p12_d,p13_d,p21_d,p22_d,p23_d,p31_d,p32_d,p33_d,p41_d,p42_d,p43_d,p51_d,p52_d,p53_d,p61_d,p62_d,p63_d,p71_d,p72_d,p73_d,p81_d,p82_d,p83_d;
wire go_conv;
wire go_conv_TOP;
wire go_dense;

reg nextstep;

reg [7:0] matrix;
wire [14:0] matrix2;    //razmer*razmer

reg [8:0] mem;
reg [8:0] filt;
reg [1:0] stride;
reg depthwise;
reg onexone;

reg [8:0] in_dense;
reg [1:0] out_dense;
reg nozero_dense;

wire clk_RAM;

wire up_perm,down_perm;
wire [SIZE_address_pix-1:0] stride_plus_prov;

conv_TOP #(
	SIZE_1,
	SIZE_2,
	SIZE_3,
	SIZE_4,
	SIZE_5,
	SIZE_6,
	SIZE_7,
	SIZE_8,
	SIZE_address_pix,
	SIZE_address_pix_t,
	SIZE_address_wei,
	SIZE_weights,
	SIZE_bias
) conv_TOP (
	.clk							(clk),
	.conv_en						(conv_en),
	.STOP							(STOP_conv),
	.memstartp					(memstartp_lvl),
	.memstartw					(memstartw_lvl),
	.memstartb					(memstartb),
	.memstartzap				(memstartzap_num),
	.read_addressp				(read_addressp_conv),
	.write_addressp			    (write_addressp_conv),
	.read_addresstp			    (read_addresstp),
	.write_addresstp			(write_addresstp),
	.read_addressb				(read_address_bias),
   .read_addressw				(read_addressw_conv),
	.we							(we_conv),
	.re_wb						(re_wb_conv),
	.re							(re_conv),
	.we_t						(we_tp),
	.re_t						(re_tp),
	.qp							(qp),
	.qtp						(qtp),
	.qw							(qw),
	.q_bias						(q_bias),
	.dp							(dp_conv),
	.dtp						(dtp),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i_to_prov					(i_conv),
	.lvl						(lvl),
	.slvl						(slvl),
	.Y1							(Y1),
	.Y2							(Y2),
	.Y3							(Y3),
	.Y4							(Y4),
	.Y5							(Y5),
	.Y6							(Y6),
	.Y7							(Y7),
	.Y8							(Y8),
	.w11							(w11_c),
	.w12							(w12_c),
	.w13							(w13_c),
	.w21							(w21_c),
	.w22							(w22_c),
	.w23							(w23_c),
	.w31							(w31_c),
	.w32							(w32_c),
	.w33							(w33_c),
	.w41							(w41_c),
	.w42							(w42_c),
	.w43							(w43_c),
	.w51							(w51_c),
	.w52							(w52_c),
	.w53							(w53_c),
	.w61							(w61_c),
	.w62							(w62_c),
	.w63							(w63_c),
	.w71							(w71_c),
	.w72							(w72_c),
	.w73							(w73_c),
	.w81							(w81_c),
	.w82							(w82_c),
	.w83							(w83_c),
	.p0_1							(p11_c),
	.p0_2							(p12_c),
	.p0_3							(p13_c),
	.p1_1							(p21_c),
	.p1_2							(p22_c),
	.p1_3							(p23_c),
	.p2_1							(p31_c),
	.p2_2							(p32_c),
	.p2_3							(p33_c),
	.p3_1							(p41_c),
	.p3_2							(p42_c),
	.p3_3							(p43_c),
	.p4_1							(p51_c),
	.p4_2							(p52_c),
	.p4_3							(p53_c),
	.p5_1							(p61_c),
	.p5_2							(p62_c),
	.p5_3							(p63_c),
	.p6_1							(p71_c),
	.p6_2							(p72_c),
	.p6_3							(p73_c),
	.p7_1							(p81_c),
	.p7_2							(p82_c),
	.p7_3							(p83_c),
	.go							(go_conv_TOP),
	.up_perm					(up_perm),
	.down_perm					(down_perm),
	.stride_plus_prov			(stride_plus_prov),
	.num						(num),
	.filt						(filt),
	.mem						(mem),
	.bias						(bias),
	.glob_average_en			(glob_average_en),
	.step						(step),
	.stride						(stride),
	.depthwise					(depthwise),
	.onexone					(onexone)
);
memorywork #(
	num_conv,
	SIZE_1,
	SIZE_2,
	SIZE_3,
	SIZE_4,
	SIZE_5,
	SIZE_6,
	SIZE_7,
	SIZE_8,
	SIZE_address_pix,
	SIZE_address_wei,
	SIZE_weights,
	SIZE_bias
) block (
	.clk_RAM_w					(clk_RAM_w),
	.we_w						(we_w),
	.re_weights					(re_weights),
	.re_bias					(re_bias),
	.load_weights				(load_weights),
	.addrw						(write_addressw),
	.dw							(dw),
	.step_out					(step),
	.nextstep					(nextstep),
	.data						(dp_weights),
	.address					(address_weights),
	.GO							(GO),
	.in_dense					(in_dense),
	.onexone					(onexone),
	.data_bias					(dp_bias),
	.load_bias					(load_bias),
	.address_bias				(address_bias),
	.write_address_bias		    (write_address_bias),
	.we_bias					(we_bias),
	.d_bias						(d_bias)
);
RAM #(
	picture_size,
	SIZE_1,
	SIZE_2,
	SIZE_3,
	SIZE_4,
	SIZE_5,
	SIZE_6,
	SIZE_7,
	SIZE_8,
	SIZE_address_pix,
	SIZE_address_pix_t,
	SIZE_address_wei,
	SIZE_address_image,
	SIZE_weights,
	SIZE_bias
) memory (
	.qp							(qp),
	.qtp						(qtp),
	.qw							(qw),
	.dp							(dp),
	.dtp						(dtp),
	.dw							(dw),
	.write_addressp			    (write_addressp),
	.read_addressp				(read_addressp),
	.write_addresstp			(write_addresstp),
	.read_addresstp			    (read_addresstp),
	.write_addressw			    (write_addressw),
	.read_addressw				(read_addressw),
	.we_p						(we_p),
	.we_tp						(we_tp),
	.we_w						(we_w),
	.re_p						(re_p),
	.re_tp						(re_tp),
	.re_w						(re_w),
	.clk						(clk_RAM),
	.clk_RAM_w					(clk_RAM_w),
	.q_bias						(q_bias),
	.d_bias						(d_bias),
	.we_bias					(we_bias),
	.re_bias					(re_bias_RAM),
	.write_address_bias		    (write_address_bias),
	.read_address_bias		    (read_address_bias)
);
border border(
	.clk						(clk),
	.go							(conv_en && (!onexone)),
	.i							(i_conv),
	.matrix						(matrix),
	.prov						(prov)
);
dense #(
	num_conv,
	SIZE_1,
	SIZE_2,
	SIZE_3,
	SIZE_4,
	SIZE_5,
	SIZE_6,
	SIZE_7,
	SIZE_8,
	SIZE_address_pix,
	SIZE_address_wei,
	SIZE_weights
) dense (
	.clk						(clk),
	.dense_en					(dense_en),
	.STOP						(STOP_dense),
	.in							(in_dense),
	.out						(out_dense),
	.we							(we_dense),
	.re_p						(re_p_dense),
	.re_w						(re_w_dense),
	.read_addressp				(read_addressp_dense),
	.read_addressw				(read_addressw_dense),
	.write_addressp			    (write_addressp_dense),
	.memstartp					(memstartp_lvl),
	.memstartzap				(memstartzap_num),
	.qp							(qp),
	.qw							(qw),
	.res						(dp_dense),
	.Y1							(Y1),
	.Y2							(Y2),
	.Y3							(Y3),
	.Y4							(Y4),
	.Y5							(Y5),
	.Y6							(Y6),
	.Y7							(Y7),
	.Y8							(Y8),
	.w11						(w11_d),
	.w12						(w12_d),
	.w13						(w13_d),
	.w21						(w21_d),
	.w22						(w22_d),
	.w23						(w23_d),
	.w31						(w31_d),
	.w32						(w32_d),
	.w33						(w33_d),
	.w41						(w41_d),
	.w42						(w42_d),
	.w43						(w43_d),
	.w51						(w51_d),
	.w52						(w52_d),
	.w53						(w53_d),
	.w61						(w61_d),
	.w62						(w62_d),
	.w63						(w63_d),
	.w71						(w71_d),
	.w72						(w72_d),
	.w73						(w73_d),
	.w81						(w81_d),
	.w82						(w82_d),
	.w83						(w83_d),
	.p11						(p11_d),
	.p12						(p12_d),
	.p13						(p13_d),
	.p21						(p21_d),
	.p22						(p22_d),
	.p23						(p23_d),
	.p31						(p31_d),
	.p32						(p32_d),
	.p33						(p33_d),
	.p41						(p41_d),
	.p42						(p42_d),
	.p43						(p43_d),
	.p51						(p51_d),
	.p52						(p52_d),
	.p53						(p53_d),
	.p61						(p61_d),
	.p62						(p62_d),
	.p63						(p63_d),
	.p71						(p71_d),
	.p72						(p72_d),
	.p73						(p73_d),
	.p81						(p81_d),
	.p82						(p82_d),
	.p83						(p83_d),
	.go							(go_dense),
	.nozero						(nozero_dense)
);
result #(
	SIZE_1,
	SIZE_2,
	SIZE_3,
	SIZE_4,
	SIZE_5,
	SIZE_6,
	SIZE_7,
	SIZE_8,
	SIZE_address_pix
) result (
	.clk						(clk),
	.enable						(result_en),
	.STOP						(STOP_res),
	.memstartp					(memstartp_lvl),
	.read_addressp				(read_addressp_res),
	.qp							(qp),
	.re							(re_p_res),
	.RESULT						(res_out)
);
conv #(
	SIZE_1,
	SIZE_address_pix,
	SIZE_weights
) conv1 (
	.clk						(clk),
	.Y1							(Y1),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i							(i_conv),
	.up_perm					((up_perm && (!dense_en))),
	.down_perm					((down_perm && (!dense_en))),
	.p1							(p11),
	.p2							(p12),
	.p3							(p13),
	.w1							(w11),
	.w2							(w12),
	.w3							(w13),
	.conv_en					(go_conv),
	.dense_en					((onexone||dense_en)),
	.stride_plus_prov			(stride_plus_prov)
);
conv #(
	SIZE_1,
	SIZE_address_pix,
	SIZE_weights
) conv2 (
	.clk						(clk),
	.Y1							(Y2),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i							(i_conv),
	.up_perm					((up_perm && (!dense_en))),
	.down_perm					((down_perm && (!dense_en))),
	.p1							(p21),
	.p2							(p22),
	.p3							(p23),
	.w1							(w21),
	.w2							(w22),
	.w3							(w23),
	.conv_en					(go_conv),
	.dense_en					((onexone||dense_en)),
	.stride_plus_prov			(stride_plus_prov)
);
conv #(
	SIZE_1,
	SIZE_address_pix,
	SIZE_weights
) conv3 (
	.clk						(clk),
	.Y1							(Y3),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i							(i_conv),
	.up_perm					((up_perm && (!dense_en))),
	.down_perm					((down_perm && (!dense_en))),
	.p1							(p31),
	.p2							(p32),
	.p3							(p33),
	.w1							(w31),
	.w2							(w32),
	.w3							(w33),
	.conv_en					(go_conv),
	.dense_en					((onexone||dense_en)),
	.stride_plus_prov			(stride_plus_prov)
);
conv #(
	SIZE_1,
	SIZE_address_pix,
	SIZE_weights
) conv4 (
	.clk						(clk),
	.Y1							(Y4),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i							(i_conv),
	.up_perm					((up_perm && (!dense_en))),
	.down_perm					((down_perm && (!dense_en))),
	.p1							(p41),
	.p2							(p42),
	.p3							(p43),
	.w1							(w41),
	.w2							(w42),
	.w3							(w43),
	.conv_en					(go_conv),
	.dense_en					((onexone||dense_en)),
	.stride_plus_prov			(stride_plus_prov)
);
conv #(
	SIZE_1,
	SIZE_address_pix,
	SIZE_weights
) conv5 (
	.clk						(clk),
	.Y1							(Y5),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i							(i_conv),
	.up_perm					((up_perm && (!dense_en))),
	.down_perm					((down_perm && (!dense_en))),
	.p1							(p51),
	.p2							(p52),
	.p3							(p53),
	.w1							(w51),
	.w2							(w52),
	.w3							(w53),
	.conv_en					(go_conv),
	.dense_en					((onexone||dense_en)),
	.stride_plus_prov			(stride_plus_prov)
);
conv #(
	SIZE_1,
	SIZE_address_pix,
	SIZE_weights
) conv6 (
	.clk						(clk),
	.Y1							(Y6),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i							(i_conv),
	.up_perm					((up_perm && (!dense_en))),
	.down_perm					((down_perm && (!dense_en))),
	.p1							(p61),
	.p2							(p62),
	.p3							(p63),
	.w1							(w61),
	.w2							(w62),
	.w3							(w63),
	.conv_en					(go_conv),
	.dense_en					((onexone||dense_en)),
	.stride_plus_prov			(stride_plus_prov)
);
conv #(
	SIZE_1,
	SIZE_address_pix,
	SIZE_weights
) conv7 (
	.clk						(clk),
	.Y1							(Y7),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i							(i_conv),
	.up_perm					((up_perm && (!dense_en))),
	.down_perm					((down_perm && (!dense_en))),
	.p1							(p71),
	.p2							(p72),
	.p3							(p73),
	.w1							(w71),
	.w2							(w72),
	.w3							(w73),
	.conv_en					(go_conv),
	.dense_en					((onexone||dense_en)),
	.stride_plus_prov			(stride_plus_prov)
);
conv #(
	SIZE_1,
	SIZE_address_pix,
	SIZE_weights
) conv8 (
	.clk						(clk),
	.Y1							(Y8),
	.prov						(prov),
	.matrix						(matrix),
	.matrix2					(matrix2),
	.i							(i_conv),
	.up_perm					((up_perm && (!dense_en))),
	.down_perm					((down_perm && (!dense_en))),
	.p1							(p81),
	.p2							(p82),
	.p3							(p83),
	.w1							(w81),
	.w2							(w82),
	.w3							(w83),
	.conv_en					(go_conv),
	.dense_en					((onexone||dense_en)),
	.stride_plus_prov			(stride_plus_prov)
);
always @(posedge clk )
begin
if (GO==1)
begin
STOP=0;
nextstep=1;
glob_average_en=0;
result_en=0;
end
else nextstep=0;
if (STOP==0)
begin
	    if ((TOPlvl==1)&&(step==3))
		    begin
			    matrix = 128;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 7;
			    filt = 2;
			    stride=2;
			    onexone=0;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==2)&&(step==3)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==2)&&(step==6))
		    begin
			    matrix = 64;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 7;
			    filt = 7;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==3)&&(step==6)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==3)&&(step==9))
		    begin
			    matrix = 64;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 7;
			    filt = 15;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==4)&&(step==9)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==4)&&(step==12))
		    begin
			    matrix = 64;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 15;
			    filt = 15;
			    stride=2;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==5)&&(step==12)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==5)&&(step==15))
		    begin
			    matrix = 32;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 15;
			    filt = 31;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==6)&&(step==15)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==6)&&(step==18))
		    begin
			    matrix = 32;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 31;
			    filt = 31;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==7)&&(step==18)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==7)&&(step==21))
		    begin
			    matrix = 32;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 31;
			    filt = 31;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==8)&&(step==21)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==8)&&(step==24))
		    begin
			    matrix = 32;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 31;
			    filt = 31;
			    stride=2;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==9)&&(step==24)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==9)&&(step==27))
		    begin
			    matrix = 16;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 31;
			    filt = 63;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==10)&&(step==27)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==10)&&(step==30))
		    begin
			    matrix = 16;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 63;
			    filt = 63;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==11)&&(step==30)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==11)&&(step==33))
		    begin
			    matrix = 16;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 63;
			    filt = 63;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==12)&&(step==33)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==12)&&(step==36))
		    begin
			    matrix = 16;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 63;
			    filt = 63;
			    stride=2;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==13)&&(step==36)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==13)&&(step==39))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 63;
			    filt = 127;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==14)&&(step==39)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==14)&&(step==42))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==15)&&(step==42)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==15)&&(step==45))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==16)&&(step==45)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==16)&&(step==48))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==17)&&(step==48)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==17)&&(step==51))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==18)&&(step==51)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==18)&&(step==54))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==19)&&(step==54)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==19)&&(step==57))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==20)&&(step==57)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==20)&&(step==60))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==21)&&(step==60)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==21)&&(step==63))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==22)&&(step==63)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==22)&&(step==66))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==23)&&(step==66)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==23)&&(step==69))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==24)&&(step==69)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==24)&&(step==72))
		    begin
			    matrix = 8;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 127;
			    stride=2;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==25)&&(step==72)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==25)&&(step==75))
		    begin
			    matrix = 4;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2;
			    conv_en = 1;
			    dense_en=0;
			    mem = 127;
			    filt = 255;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=0;
           end
	    if ((TOPlvl==26)&&(step==75)) 
           begin
               nextstep = 1;
               onexone = 0;
           end
	    if ((TOPlvl==26)&&(step==78))
		    begin
			    matrix = 4;
			    memstartp = picture_storage_limit_2;
			    memstartw = 0;
			    memstartzap = picture_storage_limit;
			    conv_en = 1;
			    dense_en=0;
			    mem = 255;
			    filt = 255;
			    stride=1;
			    onexone=0;
			    depthwise=1;
			    glob_average_en=0;
           end
	    if ((TOPlvl==27)&&(step==78)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==27)&&(step==81))
		    begin
			    matrix = 4;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2+0;
			    conv_en = 1;
			    dense_en=0;
			    mem = 255;
			    filt = 127;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=1;
           end
	    if ((TOPlvl==28)&&(step==81)) 
           begin
               nextstep = 1;
               onexone = 1;
           end
	    if ((TOPlvl==28)&&(step==84))
		    begin
			    matrix = 4;
			    memstartp = picture_storage_limit;
			    memstartw = 0;
			    memstartzap = picture_storage_limit_2+16;
			    conv_en = 1;
			    dense_en=0;
			    mem = 255;
			    filt = 127;
			    stride=1;
			    onexone=1;
			    depthwise=0;
			    glob_average_en=1;
           end
	    if ((TOPlvl==29)&&(step==84)) 
           begin
			    nextstep=1;
			    onexone=0;
			    in_dense=256;
			    out_dense=2;
			end
	    if ((TOPlvl==29)&&(step==86))
           begin
			    memstartp= picture_storage_limit_2;
			    memstartzap = picture_storage_limit;
			    conv_en=0;
			    dense_en=1;
			    nozero_dense=1;
			    depthwise=0;
			end
	    if ((TOPlvl==29)&&(STOP_dense==0)&&(step==87))
		    begin
			    memstartp = picture_storage_limit;
		    	result_en = 1;
		    end
	if ((depthwise)||(lvl==filt)||((onexone)&&(mem==((lvl+1)*8)-1))) bias=1; else bias=0;
	if ((STOP_conv)&&(conv_en==1)) conv_en=0;
	if (STOP_dense==1) begin dense_en=0; nextstep=1; end
	if ((STOP_res==1)&&(result_en==1))
	begin
		result_en=0;
		STOP=1;
	end
end
end

always @(negedge STOP_conv or posedge GO)
	begin
		if (GO)
			begin
				lvl=0;
				slvl=0;
				TOPlvl_conv=1;
				num=0;
			end
		else
			begin
				if (lvl==(filt)||((lvl==filt>>3)&&(depthwise))||((lvl==((mem+1)>>3)-1)&&(onexone)))
					begin
						lvl=0;
						if ((num!=0)&&(!depthwise)) num=num+1; else num=0;
						if ((num==0)||(depthwise))
						begin 
							if ((depthwise)||((!onexone)&&(mem==(8+(slvl*8))-1))||((onexone)&&(filt==(8+(slvl*8))-1))) 
								begin
									slvl=0; 
									TOPlvl_conv=TOPlvl_conv+1'b1;
								end
							else slvl = slvl + 1'b1;
						end
					end
				else
				lvl=lvl+1;
			end
	end

assign address_image_1 = address_image[13:0]+1;	

assign memstartw_lvl=memstartw+((onexone?num*(((mem+1)>>3)-1)+lvl:(depthwise?lvl*num_conv:lvl))+(((!depthwise)&&(!onexone))?(slvl*(4*(filt+1))):(1'b0))+((!onexone)?(num*(filt+1)):num+slvl*((mem+1)<<0)));
assign memstartzap_num = memstartzap+((glob_average_en)?(num+slvl*1):0)+(((conv_en==1)&&(!glob_average_en))?(num*((matrix>>(stride-1))*(matrix>>(stride-1)))+slvl*((matrix>>(stride-1))*(matrix>>(stride-1)))+((depthwise)?lvl*((matrix>>(stride-1))*(matrix>>(stride-1))):0)):0);
assign memstartp_lvl=memstartp+(onexone?((lvl[8:0])*matrix2):(depthwise?(lvl*matrix2):((lvl>>num_conv-1)*matrix2))); 
assign memstartb=slvl*8+num+(depthwise?lvl*num_conv:0)+1;

assign re_p=GO?1'b1:((conv_en==1)?re_conv:((dense_en==1)?re_p_dense:((result_en==1)?re_p_res:0)));
assign re_w=(conv_en==1)?re_wb_conv:((dense_en==1)?re_w_dense:0);
assign re_bias_RAM=(conv_en==1)?re_wb_conv:0;
assign read_addressp=GO?address_image_1[13:0]:((conv_en==1)?read_addressp_conv:((dense_en==1)?read_addressp_dense:((result_en==1)?read_addressp_res:0)));
assign we_p=GO?we_image:((conv_en==1)?we_conv:((dense_en==1)?we_dense:0));
assign dp=GO?((address_image<128*128*1)?{dp_image,13'd0,13'd0,13'd0,13'd0,13'd0,13'd0,13'd0}:((address_image<128*128*2)?{qp[SIZE_8-1:SIZE_7],dp_image,13'd0,13'd0,13'd0,13'd0,13'd0,13'd0}:((address_image<128*128*3)?{qp[SIZE_8-1:SIZE_7],qp[SIZE_7-1:SIZE_6],dp_image,13'd0,13'd0,13'd0,13'd0,13'd0}:0))):((conv_en==1)?dp_conv:((dense_en==1)?dp_dense:0));
assign write_addressp=GO?(address_image[13:0]):((conv_en==1)?write_addressp_conv:((dense_en==1)?write_addressp_dense:0));
assign read_addressw=(conv_en==1)?read_addressw_conv:((dense_en==1)?read_addressw_dense:0);

assign matrix2=matrix*matrix;

assign clk_RAM=GO?clk_RAM_p:clk;

assign p11=(conv_en==1)?p11_c:((dense_en==1)?p11_d:0);
assign p12=(conv_en==1)?p12_c:((dense_en==1)?p12_d:0);
assign p13=(conv_en==1)?p13_c:((dense_en==1)?p13_d:0);
assign p21=(conv_en==1)?p21_c:((dense_en==1)?p21_d:0);
assign p22=(conv_en==1)?p22_c:((dense_en==1)?p22_d:0);
assign p23=(conv_en==1)?p23_c:((dense_en==1)?p23_d:0);
assign p31=(conv_en==1)?p31_c:((dense_en==1)?p31_d:0);
assign p32=(conv_en==1)?p32_c:((dense_en==1)?p32_d:0);
assign p33=(conv_en==1)?p33_c:((dense_en==1)?p33_d:0);
assign p41=(conv_en==1)?p41_c:((dense_en==1)?p41_d:0);
assign p42=(conv_en==1)?p42_c:((dense_en==1)?p42_d:0);
assign p43=(conv_en==1)?p43_c:((dense_en==1)?p43_d:0);
assign p51=(conv_en==1)?p51_c:((dense_en==1)?p51_d:0);
assign p52=(conv_en==1)?p52_c:((dense_en==1)?p52_d:0);
assign p53=(conv_en==1)?p53_c:((dense_en==1)?p53_d:0);
assign p61=(conv_en==1)?p61_c:((dense_en==1)?p61_d:0);
assign p62=(conv_en==1)?p62_c:((dense_en==1)?p62_d:0);
assign p63=(conv_en==1)?p63_c:((dense_en==1)?p63_d:0);
assign p71=(conv_en==1)?p71_c:((dense_en==1)?p71_d:0);
assign p72=(conv_en==1)?p72_c:((dense_en==1)?p72_d:0);
assign p73=(conv_en==1)?p73_c:((dense_en==1)?p73_d:0);
assign p81=(conv_en==1)?p81_c:((dense_en==1)?p81_d:0);
assign p82=(conv_en==1)?p82_c:((dense_en==1)?p82_d:0);
assign p83=(conv_en==1)?p83_c:((dense_en==1)?p83_d:0);

assign w11=(conv_en==1)?w11_c:((dense_en==1)?w11_d:0);
assign w12=(conv_en==1)?w12_c:((dense_en==1)?w12_d:0);
assign w13=(conv_en==1)?w13_c:((dense_en==1)?w13_d:0);
assign w21=(conv_en==1)?w21_c:((dense_en==1)?w21_d:0);
assign w22=(conv_en==1)?w22_c:((dense_en==1)?w22_d:0);
assign w23=(conv_en==1)?w23_c:((dense_en==1)?w23_d:0);
assign w31=(conv_en==1)?w31_c:((dense_en==1)?w31_d:0);
assign w32=(conv_en==1)?w32_c:((dense_en==1)?w32_d:0);
assign w33=(conv_en==1)?w33_c:((dense_en==1)?w33_d:0);
assign w41=(conv_en==1)?w41_c:((dense_en==1)?w41_d:0);
assign w42=(conv_en==1)?w42_c:((dense_en==1)?w42_d:0);
assign w43=(conv_en==1)?w43_c:((dense_en==1)?w43_d:0);
assign w51=(conv_en==1)?w51_c:((dense_en==1)?w51_d:0);
assign w52=(conv_en==1)?w52_c:((dense_en==1)?w52_d:0);
assign w53=(conv_en==1)?w53_c:((dense_en==1)?w53_d:0);
assign w61=(conv_en==1)?w61_c:((dense_en==1)?w61_d:0);
assign w62=(conv_en==1)?w62_c:((dense_en==1)?w62_d:0);
assign w63=(conv_en==1)?w63_c:((dense_en==1)?w63_d:0);
assign w71=(conv_en==1)?w71_c:((dense_en==1)?w71_d:0);
assign w72=(conv_en==1)?w72_c:((dense_en==1)?w72_d:0);
assign w73=(conv_en==1)?w73_c:((dense_en==1)?w73_d:0);
assign w81=(conv_en==1)?w81_c:((dense_en==1)?w81_d:0);
assign w82=(conv_en==1)?w82_c:((dense_en==1)?w82_d:0);
assign w83=(conv_en==1)?w83_c:((dense_en==1)?w83_d:0);

assign TOPlvl=TOPlvl_conv;

assign go_conv=(conv_en==1)?go_conv_TOP:((dense_en==1)?go_dense:0);

assign RESULT=(STOP)?res_out:4'b1111;

endmodule
