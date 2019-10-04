  //sclk，sdin数据传输时序代码（i2c写控制代码）
module i2c_com(clock_i2c,          //i2c控制接口传输所需时钟，0-400khz，此处为20khz
               camera_rstn,     
               ack,              //应答信号
               i2c_data,          //sdin接口传输的32位数据
               start,             //开始传输标志
               tr_end,           //传输结束标志
               i2c_sclk,          //FPGA与camera iic时钟接口
               i2c_sdat);         //FPGA与camera iic数据接口
    input [31:0]i2c_data;
    input camera_rstn;
    input clock_i2c;
    output ack;
    input start;
    output tr_end;
    output i2c_sclk;
    inout i2c_sdat;
    reg [5:0] cyc_count;
    reg reg_sdat;
    reg sclk;
    reg ack1,ack2,ack3;
    reg tr_end;
 
   
    wire i2c_sclk;
    wire i2c_sdat;
    wire ack;
   
    assign ack=ack1|ack2|ack3;
    assign i2c_sclk=sclk|(((cyc_count>=4)&(cyc_count<=39))?~clock_i2c:0);
    assign i2c_sdat=reg_sdat?1'bz:0;
   
   
    always@(posedge clock_i2c or  negedge camera_rstn)
    begin
       if(!camera_rstn)
         cyc_count<=6'b111111;
       else 
		   begin
           if(start==0)
             cyc_count<=0;
           else if(cyc_count<6'b111111)
             cyc_count<=cyc_count+1;
         end
    end
	 
	 
    always@(posedge clock_i2c or negedge camera_rstn)
    begin
       if(!camera_rstn)
       begin
          tr_end<=0;
          ack1<=1;
          ack2<=1;
          ack3<=1;
          sclk<=1;
          reg_sdat<=1;
       end
       else
          case(cyc_count)
          0:begin ack1<=1;ack2<=1;ack3<=1;tr_end<=0;sclk<=1;reg_sdat<=1;end
          1:reg_sdat<=0;                 //开始传输
          2:sclk<=0;
          3:reg_sdat<=i2c_data[31];
          4:reg_sdat<=i2c_data[30];
          5:reg_sdat<=i2c_data[29];
          6:reg_sdat<=i2c_data[28];
          7:reg_sdat<=i2c_data[27];
          8:reg_sdat<=i2c_data[26];
          9:reg_sdat<=i2c_data[25];
          10:reg_sdat<=i2c_data[24];
          11:reg_sdat<=1;                //应答信号
          12:begin reg_sdat<=i2c_data[23];ack1<=i2c_sdat;end
          13:reg_sdat<=i2c_data[22];
          14:reg_sdat<=i2c_data[21];
          15:reg_sdat<=i2c_data[20];
          16:reg_sdat<=i2c_data[19];
          17:reg_sdat<=i2c_data[18];
          18:reg_sdat<=i2c_data[17];
          19:reg_sdat<=i2c_data[16];
          20:reg_sdat<=1;                //应答信号       
          21:begin reg_sdat<=i2c_data[15];ack1<=i2c_sdat;end
          22:reg_sdat<=i2c_data[14];
          23:reg_sdat<=i2c_data[13];
          24:reg_sdat<=i2c_data[12];
          25:reg_sdat<=i2c_data[11];
          26:reg_sdat<=i2c_data[10];
          27:reg_sdat<=i2c_data[9];
          28:reg_sdat<=i2c_data[8];
          29:reg_sdat<=1;                //应答信号       
          30:begin reg_sdat<=i2c_data[7];ack2<=i2c_sdat;end
          31:reg_sdat<=i2c_data[6];
          32:reg_sdat<=i2c_data[5];
          33:reg_sdat<=i2c_data[4];
          34:reg_sdat<=i2c_data[3];
          35:reg_sdat<=i2c_data[2];
          36:reg_sdat<=i2c_data[1];
          37:reg_sdat<=i2c_data[0];
          38:reg_sdat<=1;                //应答信号       
          39:begin ack3<=i2c_sdat;sclk<=0;reg_sdat<=0;end
          40:sclk<=1;
          41:begin reg_sdat<=1;tr_end<=1;end
          endcase
       
end
endmodule

