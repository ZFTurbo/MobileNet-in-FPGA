//camera power on timing requirement
module power_on_delay(clk_50M,reset_n,camera_rstn,camera_pwnd,initial_en);                  
input clk_50M;
input reset_n;
output camera_rstn;
output camera_pwnd;
output initial_en;
reg [18:0]cnt1;
reg [15:0]cnt2;
reg [19:0]cnt3;
reg initial_en;
reg camera_rstn_reg;
reg camera_pwnd_reg;

assign camera_rstn=camera_rstn_reg;
assign camera_pwnd=camera_pwnd_reg;

//5ms, delay from sensor power up stable to Pwdn pull down
always@(posedge clk_50M)
begin
  if(reset_n==1'b0) begin
	    cnt1<=0;
		 camera_pwnd_reg<=1'b1;  
  end
  else if(cnt1<18'h40000) begin
       cnt1<=cnt1+1'b1;
       camera_pwnd_reg<=1'b1;
  end
  else
     camera_pwnd_reg<=1'b0;         
end

//1.3ms, delay from pwdn low to resetb pull up
always@(posedge clk_50M)
begin
  if(camera_pwnd_reg==1)  begin
	    cnt2<=0;
		 camera_rstn_reg<=1'b0;  
  end
  else if(cnt2<16'hffff) begin
       cnt2<=cnt2+1'b1;
       camera_rstn_reg<=1'b0;
  end
  else
     camera_rstn_reg<=1'b1;         
end

//21ms, delay from resetb pul high to SCCB initialization
always@(posedge clk_50M)
begin
  if(camera_rstn_reg==0) begin
         cnt3<=0;
         initial_en<=1'b0;
  end
  else if(cnt3<20'hfffff) begin
        cnt3<=cnt3+1'b1;
        initial_en<=1'b0;
  end
  else
       initial_en<=1'b1;    
end

endmodule
