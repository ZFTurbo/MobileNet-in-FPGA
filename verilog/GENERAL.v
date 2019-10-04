module GENERAL(
input CLOCK_50,

//////////// ILI9341 //////////////

input 							tft_sdo, 
output  							tft_sck, 
output							tft_sdi, 
output							tft_dc, 
output							tft_reset, 
output							tft_cs,

//////////// CAMERA ///////////
output			CMOS_SCLK,	//cmos i2c clock
inout				CMOS_SDAT,	//cmos i2c data
input				CMOS_VSYNC,	//cmos vsync
input				CMOS_HREF,	//cmos hsync refrence
input				CMOS_PCLK,	//cmos pxiel clock
output			CMOS_XCLK,	//cmos externl clock
input	[7:0]	   CMOS_DB,		//cmos data
output			cmos_rst_n,	//cmos reset
output			cmos_pwdn,	//cmos pwer down

input UART_RX,
output UART_TX,
input UART_CTS,
output UART_RTS,

output		     [6:0]		HEX0,
output		     [6:0]		HEX1,

input [3:0] KEY,
output [3:0] LED
);

wire [7:0] step;

reg [15:0] x;
wire [15:0] data_camera,currentPixel;
reg [9:0] x_tft,y_tft,x_in,y_in,x_neuro_image,y_neuro_image;
reg enable_image;

//neuroset
reg GO_neuroset,start_neuroset;
reg [1:0] RESULT;
reg [2:0] step_image;
//scale_image
wire [7:0] r_out_scale,g_out_scale,b_out_scale;
wire [16:0] addr_out_scale;
wire signed [12:0] r_out_scale_13,g_out_scale_13,b_out_scale_13;
wire [15:0] ram_data;
wire valid_data_out_scale;
wire [18:0] addr_tft;
//scale_image
reg new_go,new_image,sh,sh_1,da;
wire [18:0] addr_new_image;
wire [23:0] data_scale_ram;

wire [3:0] RESULT_neuroset;
reg [3:0] res;
wire [20:0] test1;
wire [20:0] test2;
wire	test2_en;
wire signed [20:0] data_in_weights;
wire signed [20:0] data_out_weights;
reg signed [20:0] delete;
reg delete2,delete3;
wire [23:0] address_in_weights;
wire [23:0] address_out_weights;
wire [11:0] address_out_bias;

reg clk25;
wire clk100;
reg  [1:0]           pre_button;
reg                  trigger;
reg start_wr;

reg [1:0] sh_result;
reg RESULT_0,RESULT_1,RESULT_2;
// / //////////////////////////////////////////////
// reset_n and start_n control
reg [31:0]  cont;
always@(posedge CLOCK_50)
cont<=(cont==32'd4_000_001)?32'd0:cont+1'b1;

reg[4:0] sample;
always@(posedge CLOCK_50)
begin
	if(cont==32'd4_000_000)
		sample[4:0]={sample[3:0],KEY[0] || KEY[3]};
	else 
		sample[4:0]=sample[4:0];
end

assign test_software_reset_n=(sample[1:0]==2'b10)?1'b0:1'b1;
assign test_global_reset_n   =(sample[3:2]==2'b10)?1'b0:1'b1;
assign test_start_n         =(sample[4:3]==2'b01)?1'b0:1'b1;

//////////////////////////////////////////////////
pll_24_100 pll_24_100
(
	.refclk	(CLOCK_50),
	.rst		(1'b0/*~rst_n*/),
	.locked	(),
			
	.outclk_0      (clk24),                //24Mhz
	.outclk_1      (clk100),               //100Mhz
	.outclk_2		(clk60)

);

always @(posedge CLOCK_50 or negedge KEY[0]) 
if (!KEY[0]) 
	begin
		clk25=0;
		pre_button <= 2'b11;
		trigger <= 1'b0;
	end
else 
	begin
		clk25=!clk25;
		pre_button <= {pre_button[0], test_start_n};
		trigger <= !pre_button[0] && pre_button[1];
		if (trigger) start_wr=1;
	end

ov5640 ov5640
(
	.CLOCK			(CLOCK_50),	
	.rst_n			(KEY[0]), 
	.clk24			(clk24),
	
	.CMOS_SCLK		(CMOS_SCLK),	//cmos i2c clock
	.CMOS_SDAT		(CMOS_SDAT),	//cmos i2c data
	.CMOS_VSYNC		(CMOS_VSYNC),	//cmos vsync
	.CMOS_HREF		(CMOS_HREF),	//cmos hsync refrence
	.CMOS_PCLK		(CMOS_PCLK),	//cmos pxiel clock
	.CMOS_XCLK		(CMOS_XCLK),	//cmos externl clock
	.CMOS_DB			(CMOS_DB),	//cmos data
	.cmos_rst_n		(cmos_rst_n),	//cmos reset
	.cmos_pwdn		(cmos_pwdn),	//cmos pwer down
	
	.sys_we			(clk_camera),			//system data write enable
	.sys_data_in	(data_camera),       //system data input
	.frame_valid	(valid_data_camera)	//data valid, or address restart
);
always @(posedge CMOS_PCLK or negedge KEY[0])
	begin
		if (!KEY[0])	
			begin
				new_go=0;
				new_image=0;
				x_in=0;
				y_in=0;
				sh=0;
				sh_1=0;
				da=0;
			end
		else	
		begin
		if ((!GO_neuroset) && da) sh=sh+1;
		da=GO_neuroset;
		if (((new_image==0)&&(new_go==0)&&(!GO_neuroset)&&(sh_1!=sh))) 
			begin
				new_go = 1'b1;
				sh_1=sh;
			end
		if ((new_image==0)&&(new_go==1)&&(x_in==0)&&(y_in==0)) 
			begin
				new_go = 0;
				new_image = 1;
			end
		if ((clk_camera)&&(valid_data_camera)) 
					begin
						if (x_in < (320-1)) x_in = x_in+1'b1;
						else 
							begin
								x_in = 0;
								if (y_in < 240-1) y_in = y_in+1'b1;
								else
									begin
										y_in=0;
										new_image=0;
									end
							end
					end
		if (!valid_data_camera) 
			begin
				x_in=0;
			end
		end
end
assign addr_new_image=x_in+y_in*320;

tft_ili9341 #(.INPUT_CLK_MHZ(100)) tft(
	.clk					(clk100), 
	.tft_sdo				(tft_sdo), 
	.tft_sck				(tft_sck), 
	.tft_sdi				(tft_sdi), 
	.tft_dc				(tft_dc), 
	.tft_reset			(tft_reset), 
	.tft_cs				(tft_cs), 
	.framebufferData	(((x_tft<20)&&(y_tft<20))?(RESULT[0]?16'b1110000000000:16'b100):{currentPixel[7],currentPixel[6],currentPixel[5],currentPixel[4],currentPixel[3],currentPixel[2],currentPixel[1],currentPixel[0],currentPixel[15],currentPixel[14],currentPixel[13],currentPixel[12],currentPixel[11],currentPixel[10],currentPixel[9],currentPixel[8]}), 
	.framebufferClk	(fbClk)
);

always @(posedge fbClk or negedge KEY[0])
	if (!KEY[0])
		begin
			x_tft=0;
			y_tft=0;
		end
	else
		begin
			if (x_tft<319) x_tft=x_tft+1;
			else
				begin
					x_tft=0;
					if (y_tft<239) y_tft=y_tft+1;
					else y_tft=0;
				end
		end
		
always @(posedge CLOCK_50 or negedge KEY[0])
	if (!KEY[0])
		begin
			x_neuro_image=0;
			y_neuro_image=0;
		end
	else
		begin
			if (GO_neuroset)
				begin
					if (x_neuro_image<128-1) x_neuro_image=x_neuro_image+1;
					else
						begin
							x_neuro_image=0;
							if (y_neuro_image<128-1) y_neuro_image=y_neuro_image+1;
							else y_neuro_image=0;
						end
				end
		end
		
always @(posedge CLOCK_50 or negedge KEY[0])
	begin
		if (!KEY[0])
			begin
				start_neuroset = 0;
				GO_neuroset = 0;
				step_image = 0;
				RESULT = 0;
			end
		else
			begin
				if (STOP_neuroset) 
					begin
						step_image = 1;
						if (sh_result<2) sh_result = sh_result+1'b1;
						else sh_result=0;
						if (sh==0) RESULT_0 = RESULT_neuroset[1:0];
						if (sh==1) RESULT_1 = RESULT_neuroset[1:0];
						if (sh==2) RESULT_2 = RESULT_neuroset[1:0];
						if (sh_result==0) 
							begin
								if (RESULT_0==RESULT_1) RESULT = RESULT_0;
								else if (RESULT_0==RESULT_2) RESULT = RESULT_0;
								else if (RESULT_1==RESULT_2) RESULT = RESULT_1;
							end
					end
				if (UART_stop) start_neuroset = 1;
				if ((start_neuroset)&&(step_image<4))
					begin
						if ((y_neuro_image*128+x_neuro_image) == 0) step_image = step_image + 1;
						GO_neuroset = 1;
					end
				else 
					GO_neuroset = 0;
			end
	end
assign addr_tft = x_tft + y_tft*320;

serialGPIO(
    .clk25		(clk25),
    .RxD			(UART_RX),
    .TxD			(UART_TX),
	 .reset		(KEY[0]),
	 
	 .address	(address_in_weights),
	 .data		(data_in_weights),
	 .write_enable	(we_weights),
	 .start		(UART_start),
	 .stop		(UART_stop),
	 
	 .data_tx	(test2[15:8]),
	 .enable_tx	(test2_en && delete2)
);

scale_picture scale_picture(
	.clk					(fbClk),
	.rst					(KEY[0]),
	.valid_data			(1), 
	.r						({currentPixel[4],currentPixel[3],currentPixel[2],currentPixel[1],currentPixel[0],1'b0,1'b0,1'b0}),
	.g						({currentPixel[10],currentPixel[9],currentPixel[8],currentPixel[7],currentPixel[6],currentPixel[5],1'b0,1'b0}),
	.b						({currentPixel[15],currentPixel[14],currentPixel[13],currentPixel[12],currentPixel[11],1'b0,1'b0,1'b0}),
	.x						(x_tft),
	.y						(y_tft),  
	.r_out				(r_out_scale),
	.g_out				(g_out_scale),
	.b_out				(b_out_scale),
	.addr_out			(addr_out_scale),
	.valid_data_out	(valid_data_out_scale)
);
		
assign r_out_scale_13 = (data_scale_ram[23:16]*2 - 255)*16;
assign g_out_scale_13 = (data_scale_ram[15:8]*2 - 255)*16;
assign b_out_scale_13 = (data_scale_ram[7:0]*2 - 255)*16;


TOP neuroset(
.clk					(CLOCK_50),
.clk_RAM_w			(CLOCK_50), 
.clk_RAM_p			(CLOCK_50),  
.GO					(GO_neuroset),
.RESULT				(RESULT_neuroset),
.STOP					(STOP_neuroset),

.re_weights			(re_weights),
.load_weights		(re_weights && (!GO_neuroset)), 
.dp_weights			(data_out_weights),
.address_weights	(address_out_weights),

.re_bias				(re_bias),
.load_bias			(re_bias),
.dp_bias				(data_out_weights),
.address_bias		(address_out_bias),

.we_image			(GO_neuroset),
.dp_image			((step_image==1)?(b_out_scale_13):((step_image==2)?(g_out_scale_13):((step_image==3)?(r_out_scale_13):0))), 
.address_image		((x_neuro_image+y_neuro_image*128)+((step_image-1)*128*128)),

.step					(step)/*,
.test1				(test1),
.test2				(test2),
.test2_en			(test2_en)*/
);

RAM_general RAM_general(
.clk_in (clk25),
.clk_out (CLOCK_50),
.clk_in_im (CMOS_PCLK),
.clk_out_im (fbClk),
.clk_in_im_scale (fbClk),
.clk_out_im_scale (CLOCK_50),

.data_in_im (data_camera),
.data_out_im (currentPixel),
.address_in_im (addr_new_image),
.address_out_im (addr_tft),
.we_image ((new_image)&&(clk_camera)&&(valid_data_camera)),
.re_image (1),

.data_in_im_scale ({r_out_scale,g_out_scale,b_out_scale}),
.data_out_im_scale (data_scale_ram),
.address_in_im_scale (addr_out_scale),
.address_out_im_scale (x_neuro_image+y_neuro_image*128),
.we_image_scale (valid_data_out_scale),
.re_image_scale (GO_neuroset),

.data_in_weights (data_in_weights),
.data_out_weights (data_out_weights),
.address_in_weights (address_in_weights),
.address_out_weights ((re_weights)?address_out_weights:(208115 + address_out_bias)),
.we_weights (we_weights),
.re_weights	((re_weights && (!GO_neuroset)) || re_bias),

);

reg [31:0] sh_cadr,res_sh_cadr;
reg wait_cadr;
reg [2:0] sh_show;

assign LED[3] = new_image;
assign LED[2] = sh;
assign LED[1] = sh_1;
assign LED[0] = sh_show[0];

always @(posedge CLOCK_50 or negedge KEY[0])
if (!KEY[0])
	begin
		wait_cadr=0;
		sh_cadr=0;
		res_sh_cadr=0;
	end
else
	begin
		if (GO_neuroset && (!STOP_neuroset) && (!wait_cadr)) 
			begin
				sh_cadr=0;
				wait_cadr=1;
			end
		else sh_cadr=sh_cadr+1;
		if (/*!GO_neuroset*/STOP_neuroset && wait_cadr) 
			begin
				res_sh_cadr=sh_cadr;
				wait_cadr=0;
			end
	end
	
always @(negedge KEY[3] or negedge KEY[0])
if (!KEY[0])	sh_show=0;
else	if (sh_show<3) sh_show=sh_show+1;
		else sh_show=0;

Seg7 seg7_0(
	.data		(((sh_show==0)?res_sh_cadr[3:0]:((sh_show==1)?res_sh_cadr[11:8]:((sh_show==2)?res_sh_cadr[19:16]:((sh_show==3)?res_sh_cadr[27:24]:0))))),
	.hex		(HEX0)
);

Seg7 seg7_1(
	.data		(((sh_show==0)?res_sh_cadr[7:4]:((sh_show==1)?res_sh_cadr[15:12]:((sh_show==2)?res_sh_cadr[23:20]:((sh_show==3)?res_sh_cadr[31:28]:0))))),
	.hex		(HEX1)
);

endmodule
