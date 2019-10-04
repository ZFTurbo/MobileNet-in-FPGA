/*-------------------------------------------------------------------------
Description			:		sdram test with uart interface.
Modification History	:
Data			By			Version			Change Description
===========================================================================

--------------------------------------------------------------------------*/

`timescale 1ns/1ns
module CMOS_Capture
(
	//Global Clock
	input				iCLK,			//24MHz
	input				iRST_N,

	//I2C Initilize Done
	input				Init_Done,		//Init Done
	
	//Sensor Interface
	output				CMOS_XCLK,		//24MHz
	input				CMOS_PCLK,		//24MHz
	input	[7:0]		CMOS_iDATA,		//CMOS Data
	input				CMOS_VSYNC,		//L: Vaild
	input				CMOS_HREF,		//H: Vaild
	
	//Ouput Sensor Data
	output	reg			CMOS_oCLK,		//1/2 PCLK
	output	reg	[15:0]	CMOS_oDATA,		//16Bits RGB		
	output	reg			CMOS_VALID,		//Data Enable
	output	reg	[7:0]	CMOS_FPS_DATA	//cmos fps
);

assign	CMOS_XCLK = iCLK;		//24MHz XCLK


//-----------------------------------------------------
//同步输入//Sensor HS & VS Vaild Capture
/**************************************************
________							       ________
VS		|_________________________________|
HS			  _______	 	   _______
_____________|       |__...___|       |____________
**************************************************/

/*
//----------------------------------------------
reg		mCMOS_HREF;		//行同步：高电平有效
always@(posedge CMOS_PCLK or negedge iRST_N)
begin
	if(!iRST_N)
		mCMOS_HREF <= 0;
	else
		mCMOS_HREF <= CMOS_HREF;		
end
wire	CMOS_HREF_over = ({mCMOS_HREF,CMOS_HREF} == 2'b10) ? 1'b1 : 1'b0;		//HREF 下降沿结束
*/

//----------------------------------------------
reg		mCMOS_VSYNC;
always@(posedge CMOS_PCLK or negedge iRST_N)
begin
	if(!iRST_N)
		mCMOS_VSYNC <= 1;
	else
		mCMOS_VSYNC <= CMOS_VSYNC;		//场同步：低电平有效
end
wire	CMOS_VSYNC_over = ({mCMOS_VSYNC,CMOS_VSYNC} == 2'b01) ? 1'b1 : 1'b0;	//VSYNC上升沿结束


/*
//--------------------------------------------
//Counter the HS & VS Pixel
localparam		H_DISP	=	12'd640;
localparam		V_DISP	=	12'd480;
reg		[11:0]	X_Cont;	//640
reg		[11:0]	Y_Cont;	//480
always@(posedge CMOS_PCLK or negedge iRST_N)
begin
	if(!iRST_N)
		X_Cont <= 0;
	else if(~CMOS_VSYNC & CMOS_HREF)			//场信号有效
		X_Cont <= (byte_state == 1'b1) ?  X_Cont + 1'b1 : X_Cont;
	else
		X_Cont <= 0;
end

always@(posedge CMOS_PCLK or negedge iRST_N)
begin
	if(!iRST_N)
		Y_Cont <= 0;
	else if(CMOS_VSYNC == 1'b0)
		begin
		if(CMOS_HREF_over == 1'b1)		//HREF下降沿 一行结束
			Y_Cont <= Y_Cont + 1'b1;
		end
	else
		Y_Cont <= 0;
end
*/

//-----------------------------------------------------
//Change the sensor data from 8 bits to 16 bits.
reg			byte_state;		//byte state count
reg [7:0]  	Pre_CMOS_iDATA;
always@(posedge CMOS_PCLK or negedge iRST_N)
begin
	if(!iRST_N)
		begin
		byte_state <= 0;
		Pre_CMOS_iDATA <= 8'd0;
		CMOS_oDATA <= 16'd0;
		end
	else
		begin
		if(~CMOS_VSYNC & CMOS_HREF)			//行场有效，{first_byte, second_byte} 
			begin
			byte_state <= byte_state + 1'b1;	//（RGB565 = {first_byte, second_byte}）
			case(byte_state)
			1'b0 :	Pre_CMOS_iDATA[7:0] <= CMOS_iDATA;
			1'b1 : 	CMOS_oDATA[15:0] <= {Pre_CMOS_iDATA[7:0], CMOS_iDATA[7:0]};
			endcase
			end
		else
			begin
			byte_state <= 0;
			Pre_CMOS_iDATA <= 8'd0;
			CMOS_oDATA <= CMOS_oDATA;
			end
		end
end


//--------------------------------------------
//Wait for Sensor output Data valid， 10 Franme
reg	[3:0] 	Frame_Cont;
reg 		Frame_valid;
always@(posedge CMOS_PCLK or negedge iRST_N)
begin
	if(!iRST_N)
		begin
		Frame_Cont <= 0;
		Frame_valid <= 0;
		end
	else if(Init_Done)					//CMOS I2C初始化完毕
		begin
		if(CMOS_VSYNC_over == 1'b1)		//VS上升沿，1帧写入完毕
			begin
			if(Frame_Cont < 12)
				begin
				Frame_Cont	<=	Frame_Cont + 1'b1;
				Frame_valid <= 1'b0;
				end
			else
				begin
				Frame_Cont	<=	Frame_Cont;
				Frame_valid <= 1'b1;		//数据输出有效
				end
			end
		end
end

//-----------------------------------------------------
//CMOS_DATA数据同步输出使能时钟
always@(posedge CMOS_PCLK or negedge iRST_N)
begin
	if(!iRST_N)
		CMOS_oCLK <= 0;
	else if(Frame_valid == 1'b1 && byte_state)//(X_Cont >= 12'd1 && X_Cont <= H_DISP))
		CMOS_oCLK <= ~CMOS_oCLK;
	else
		CMOS_oCLK <= 0;
end

//----------------------------------------------------
//数据输出有效CMOS_VALID
always@(posedge CMOS_PCLK or negedge iRST_N)
begin
	if(!iRST_N)
		CMOS_VALID <= 0;
	else if(Frame_valid == 1'b1)
		CMOS_VALID <= ~CMOS_VSYNC;
	else
		CMOS_VALID <= 0;
end


/************************************************************
	Caculate Frame Rate per second
*************************************************************/
//-----------------------------------------------------
//	2s 延时函数
reg	[25:0]	delay_cnt;	//25_000000 * 2
always@(posedge iCLK or negedge iRST_N)
begin
	if(!iRST_N)
		delay_cnt <= 0;
	else if(Frame_valid)
		begin
		if(delay_cnt < 26'd50_000000)
			delay_cnt <= delay_cnt + 1'b1;
		else
			delay_cnt <= 0;
		end
	else
		delay_cnt <= 0;
end
wire	delay_2s = (delay_cnt == 26'd50_000000) ? 1'b1 : 1'b0;

//-------------------------------------------
//帧率采样计算
reg			fps_state;
reg	[7:0]	fps_data;
always@(posedge iCLK or negedge iRST_N)
begin
	if(!iRST_N)
		begin
		fps_data <= 0;
		fps_state <= 0;
		CMOS_FPS_DATA <= 0;
		end
	else if(Frame_valid)
		begin
		case(fps_state)
		0:	begin
			CMOS_FPS_DATA <= CMOS_FPS_DATA;
			if(delay_2s == 0)
				begin
				fps_state <= 0;
				if(CMOS_VSYNC_over == 1'b1)		//VS上升沿，1帧写入完毕
					fps_data <= fps_data + 1'b1;
				end
			else
				fps_state <= 1;
			end
		1:	begin
			fps_state <= 0;
			fps_data <= 0;
			CMOS_FPS_DATA <= fps_data >>1;
			end
		endcase
		end
	else
		begin
		fps_data <= 0;
		fps_state <= 0;
		CMOS_FPS_DATA <= 0;
		end
end


endmodule



