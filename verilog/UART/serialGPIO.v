module serialGPIO(
    input clk25,
    input RxD,
    output TxD,
	 
	 input reset,
	 output	reg [23:0] address,
	 output  reg signed [20:0] data,
	 output  reg write_enable,
	 output  reg start,
	 output  reg stop,
	 
	 output RxD_data_ready,
	 
	 input signed [20:0] data_tx,
	 input enable_tx
	 
);

reg [7:0] GPout;

reg [1:0] sh;

wire [7:0] RxD_data;

async_receiver RX(.clk(clk25), .RxD(RxD), .RxD_data_ready(RxD_data_ready), .RxD_data(RxD_data));
always @(posedge clk25) if(RxD_data_ready) GPout <= RxD_data;

async_transmitter TX(.clk(clk25), .TxD(TxD), .TxD_start(enable_tx), .TxD_data(data_tx[7:0]));

always @(posedge RxD_data_ready or negedge reset)
	if (!reset)
	begin
			address = -1;
			start=0;
			sh=0;
			write_enable=0;
			data=0;
			stop=0;
	end 
	else
	begin
	

		if ((!(GPout == 255))&&(sh==0)) address = address + 1'b1;
		
		if (GPout == 191) 
			begin
				start = 0;
				stop = 1;
				sh=0;
				write_enable=0;
			end
	
		if (start)
			begin		
				if (sh==0) 
					begin
						data=0;
						data[5:0]=GPout[5:0];
						write_enable=0;
					end
				if (sh==1) 
					begin
						data[11:6]=GPout[5:0];
					end
				if (sh==2) 
					begin
						data[17:12]=GPout[5:0];
					end
				if (sh==3) 
					begin
						data[19:18]=GPout[1:0];  //minus,data
						if (GPout[3]) data=-data;
						write_enable = 1'b1;
					end
				sh=sh+1;
			end
			
		if (GPout == 255) 
			begin
				address = -1;
				start=1;
				sh=0;
				write_enable=0;
				data=0;
				stop=0;
			end
	end
	
endmodule 