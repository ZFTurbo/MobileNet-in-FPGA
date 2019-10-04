/*-------------------------------------------------------------------------
Description			:		sdram vga controller with ov7670 display.
===========================================================================
15/02/1
--------------------------------------------------------------------------*/
`timescale 1 ns / 1 ns
module system_ctrl
(
	input 		clk,		//50MHz
	input 		rst_n,		//global reset

	output 		sys_rst_n,	//system reset		
	input 		clk_c1,
	output		clk_c2,	//-75deg
	output		clk_c3	//-75deg

);

//----------------------------------------------
reg  [9:0]   delay_cnt;
reg  delay_done;
always @(posedge clk_c1 or negedge rst_n)
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
//----------------------------------------------
//Component instantiation
wire clk_50;
wire clk_c2_oddr,clk_c0_oddr;	


/*
ODDR2 #(
    .DDR_ALIGNMENT("NONE"), // Sets output alignment to "NONE", "C0" or "C1" 
    .INIT(1'b0),    // Sets initial state of the Q output to 1'b0 or 1'b1
    .SRTYPE("SYNC") // Specifies "SYNC" or "ASYNC" set/reset
    ) U_ODDR2_c2
(
      .Q(clk_c2),   // 1-bit DDR output data
      .C0(clk_c2_oddr),   // 1-bit clock input
      .C1(~clk_c2_oddr),   // 1-bit clock input
      .CE(1'b1), // 1-bit clock enable input
      .D0(1'b1), // 1-bit data input (associated with C0)
      .D1(1'b0), // 1-bit data input (associated with C1)
      .R(1'b0),   // 1-bit reset input
      .S(1'b0)    // 1-bit set input
);*/

/*
ODDR2 #(
    .DDR_ALIGNMENT("NONE"), // Sets output alignment to "NONE", "C0" or "C1" 
    .INIT(1'b0),    // Sets initial state of the Q output to 1'b0 or 1'b1
    .SRTYPE("SYNC") // Specifies "SYNC" or "ASYNC" set/reset
    ) U_ODDR2_c0
(
      .Q(clk_c0),   // 1-bit DDR output data
      .C0(clk_c0_oddr),   // 1-bit clock input
      .C1(~clk_c0_oddr),   // 1-bit clock input
      .CE(1'b1), // 1-bit clock enable input
      .D0(1'b1), // 1-bit data input (associated with C0)
      .D1(1'b0), // 1-bit data input (associated with C1)
      .R(1'b0),   // 1-bit reset input
      .S(1'b0)    // 1-bit set input
);
*/

endmodule
