/*-------------------------------------------------------------------------
Filename			:		sdram_ov5640_vga.v
Description			:		sdram vga controller with ov5640 display 1024 * 768.
Modification History	:
Data			By			Version			Change Description
===========================================================================
15/02/1
--------------------------------------------------------------------------*/
`timescale 1ns / 1ps
module ov5640
(
	//global clock 50MHz
	input			CLOCK,	
	input			rst_n,          //global reset
	
	/*//sdram control
	output			S_CLK,		//sdram clock
	output			S_CKE,		//sdram clock enable
	output			S_NCS,		//sdram chip select
	output			S_NWE,		//sdram write enable
	output			S_NCAS,	        //sdram column address strobe
	output			S_NRAS,	        //sdram row address strobe
	output  [1 :0] 	        S_DQM,		//sdram data enable 
	output	[1 :0]	        S_BA,		//sdram bank address
	output	[12:0]	        S_A,		//sdram address
	inout	[15:0]	        S_DB,		//sdram data
	
	//VGA port			
	output			VGA_HSYNC,      //horizontal sync 
	output			VGA_VSYNC,      //vertical sync
	output	[15:0]	lcd_rgb,		//VGA data*/
	
	//cmos interface
	output			CMOS_SCLK,	//cmos i2c clock
	inout			CMOS_SDAT,	//cmos i2c data
	input			CMOS_VSYNC,	//cmos vsync
	input			CMOS_HREF,	//cmos hsync refrence
	input			CMOS_PCLK,	//cmos pxiel clock
	output			CMOS_XCLK,	//cmos externl clock
	input	[7:0]	        CMOS_DB,	//cmos data
	output			cmos_rst_n,	//cmos reset
	output			cmos_pwdn,	//cmos pwer down
	
	output		sys_we,			//system data write enable
	output	[15:0]          sys_data_in,            //system data input
	output			frame_valid,		//data valid, or address restart

        input                   KEY1,           //KEY1 input
	output	[3:0]	        LED,		//led data input	
	input clk24
);

//---------------------------------------------
wire                    clk_vga;		//vga clock
wire                    clk_ref;		//sdram ctrl clock
wire                    clk_refout;		//sdram clock output
wire                    clk_25M;
                        
wire                    sys_rst_n;		//global reset
                        
wire			sdram_init_done;        //sdram init done


system_ctrl	u_system_ctrl
(
	.clk				(CLOCK),        //global clock  50MHZ
	.rst_n				(rst_n),        //external reset
	
	.sys_rst_n			(sys_rst_n),	//global reset
	.clk_c1				(CLOCK)

);
 

wire initial_en; 
wire Config_Done;
 
 //�ϵ��ӳٲ���
power_on_delay	power_on_delay_inst(
	.clk_50M                 (clk24),
	.reset_n                 (sys_rst_n),	
	.camera_rstn             (cmos_rst_n),
	.camera_pwnd             (cmos_pwdn),
	.initial_en              (initial_en)		
);

//Camera��ʼ������,Camera LED FLASH control
reg_config	reg_config_inst(
	.clk_25M                 (clk24),
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
wire	[7:0]           cmos_fps_data;		//cmos frame rate
CMOS_Capture	u_CMOS_Capture
(
	//Global Clock
	.iCLK				(clk24),		//24MHz
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
	.CMOS_FPS_DATA		        ()//(cmos_fps_data)		//cmos frame rate
);



endmodule
