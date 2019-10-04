module camera_contoller(
	output			CMOS_SCLK,	//cmos i2c clock
	inout				CMOS_SDAT,	//cmos i2c data
	input				CMOS_VSYNC,	//cmos vsync
	input				CMOS_HREF,	//cmos hsync refrence
	input				CMOS_PCLK,	//cmos pxiel clock
	output			CMOS_XCLK,	//cmos externl clock
	input	[7:0]	   CMOS_DB,		//cmos data
	output			cmos_rst_n,	//cmos reset
	output			cmos_pwdn,	//cmos pwer down
);

wire                    clk_camera;
wire                    clk_vga;		//vga clock
wire                    clk_ref;		//sdram ctrl clock
wire                    clk_refout;		//sdram clock output
wire                    clk_25M;
                        
wire                    sys_rst_n;		//global reset
                        
wire			sys_we;			//system data write enable
wire	[15:0]          sys_data_in;            //system data input
wire			sdram_init_done;        //sdram init done
 

wire initial_en; 
wire locked;


//pll pll_inst(clk, tft_clk, clk_10khz, clk_camera, locked);
pll_test pll_test ( .refclk	(clk),   //  refclk.clk
			  .rst		(1'b0),      //   reset.reset
			  .outclk_0	(tft_clk), // outclk0.clk
			  .outclk_1 (), // outclk1.clk
			  .outclk_2 (clk_camera), // outclk2.clk
			  .locked	(locked)    //  locked.export
	);

reg  [9:0]   delay_cnt;
reg  delay_done;

always @(posedge tft_clk or negedge rst_n)
begin
	if(!rst_n)
		begin
		delay_cnt <= 0;
		delay_done <= 1'b0;
		end
	else
		begin
		  if (delay_cnt== 1000)
			 delay_done <= 1'b1;
        else
          delay_cnt <= delay_cnt +1'b1;
		end
end

assign sys_rst_n=delay_done;
 
 //上电延迟部分
power_on_delay	power_on_delay_inst(
	.clk_50M                 (clk_camera),
	.reset_n                 (sys_rst_n),	
	.camera_rstn             (cmos_rst_n),
	.camera_pwnd             (cmos_pwdn),
	.initial_en              (initial_en)		
);

//Camera初始化部分,Camera LED FLASH control
reg_config	reg_config_inst(
	.clk_25M                 (clk_camera),
	.camera_rstn             (cmos_rst_n),
	.initial_en              (initial_en),		
	.i2c_sclk                (CMOS_SCLK),
	.i2c_sdat                (CMOS_SDAT),
	.reg_conf_done           (Config_Done),
	.strobe_flash            (),
	.reg_index               (),
	.clock_20k               (),
	.key1                    (KEY1)
);
 
//-----------------------------------------------               
wire			frame_valid;		//data valid, or address restart
wire	[7:0]           cmos_fps_data;		//cmos frame rate
CMOS_Capture	u_CMOS_Capture
(
	//Global Clock
	.iCLK				(clk_camera),		//24MHz
	.iRST_N				(sys_rst_n),	//global reset
	
	//I2C Initilize Done
	.Init_Done			(Config_Done /*& sdram_init_done*/),	//Init Done
	
	//Sensor Interface
	.CMOS_XCLK			(CMOS_XCLK),		//cmos
	.CMOS_PCLK			(CMOS_PCLK),		//24MHz
	.CMOS_iDATA			(CMOS_DB),    	//CMOS Data
	.CMOS_VSYNC			(CMOS_VSYNC),  	 	//L: Vaild
	.CMOS_HREF			(CMOS_HREF), 		//H: Vaild
	                                    
	//Ouput Sensor Data                 
	.CMOS_oCLK			(sys_we),			//Data PCLK
	.CMOS_oDATA			(sys_data_in),  	//16Bits RGB
	.CMOS_VALID			(frame_valid),		//Data Enable
	.CMOS_FPS_DATA		({LED[7],LED[6],LED[5],LED[4],LED[3],LED[2],LED[1],LED[0]})//(cmos_fps_data)		//cmos frame rate
);

endmodule
