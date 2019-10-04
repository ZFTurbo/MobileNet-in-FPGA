module addressRAM(
	input [6:0] step,
	output reg re_weights,
	output reg re_bias,
	output reg [17:0] firstaddr, lastaddr
);
parameter convolution_size = 9;
parameter conv1 = 1*8*3 * convolution_size;
parameter conv2_1 = 8 * convolution_size + conv1;
parameter conv2_2 = (8*8*2) + conv2_1;
parameter conv3_1 = 16 * convolution_size + conv2_2;
parameter conv3_2 = (16*16*2) + conv3_1;
parameter conv4_1 = 32 * convolution_size + conv3_2;
parameter conv4_2 = (32*32) + conv4_1;
parameter conv5_1 = 32 * convolution_size + conv4_2;
parameter conv5_2 = (32*32*2) + conv5_1;
parameter conv6_1 = 64 * convolution_size + conv5_2;
parameter conv6_2 = (64*64) + conv6_1;
parameter conv7_1 = 64 * convolution_size + conv6_2;
parameter conv7_2 = (64*64*2) + conv7_1;
parameter conv8_1 = 128 * convolution_size + conv7_2;
parameter conv8_2 = (128*128) + conv8_1;
parameter conv9_1 = 128 * convolution_size + conv8_2;
parameter conv9_2 = (128*128) + conv9_1;
parameter conv10_1 = 128 * convolution_size + conv9_2;
parameter conv10_2 = (128*128) + conv10_1;
parameter conv11_1 = 128 * convolution_size + conv10_2;
parameter conv11_2 = (128*128) + conv11_1;
parameter conv12_1 = 128 * convolution_size + conv11_2;
parameter conv12_2 = (128*128) + conv12_1;
parameter conv13_1 = 128 * convolution_size + conv12_2;
parameter conv13_2 = (128*128*2) + conv13_1;
parameter conv14_1 = 256 * convolution_size + conv13_2;
parameter conv14_2_1 = ((256*256)>>1) + conv14_1;
parameter conv14_2_2 = ((256*256)>>1) + conv14_2_1;
parameter predict = 512 + conv14_2_2;


parameter bias1 = 8;
parameter bias2_1 = (8)+8;
parameter bias2_2 = (16)+16;
parameter bias3_1 = (32)+16;
parameter bias3_2 = (48)+32;
parameter bias4_1 = (80)+32;
parameter bias4_2 = (112)+32;
parameter bias5_1 = (144)+32;
parameter bias5_2 = (176)+64;
parameter bias6_1 = (240)+64;
parameter bias6_2 = (304)+64;
parameter bias7_1 = (368)+64;
parameter bias7_2 = (432)+128;
parameter bias8_1 = (560)+128;
parameter bias8_2 = (688)+128;
parameter bias9_1 = (816)+128;
parameter bias9_2 = (944)+128;
parameter bias10_1 = (1072)+128;
parameter bias10_2 = (1200)+128;
parameter bias11_1 = (1328)+128;
parameter bias11_2 = (1456)+128;
parameter bias12_1 = (1584)+128;
parameter bias12_2 = (1712)+128;
parameter bias13_1 = (1840)+128;
parameter bias13_2 = (1968)+256;
parameter bias14_1 = (2224)+256;
parameter bias14_2_1 = (2480)+(256>>1);
parameter bias14_2_2 = (2608)+(256>>1);


always @(step)
case (step) 
8'd1: begin       //weights conv1 
		firstaddr = 0;
		lastaddr = conv1;
		re_weights = 1;
		re_bias = 0;
	  end
8'd2: begin	//bias conv1
		firstaddr = 0;
		lastaddr = bias1;
		re_weights = 0;
		re_bias = 1;
      end
8'd4: begin  //weights conv2 dw 1
		firstaddr = conv1;
		lastaddr = conv2_1;
		re_weights = 1;
		re_bias = 0;
	  end
8'd5: begin	//bias conv2 dw
		firstaddr = bias1;
		lastaddr = bias2_1;
		re_weights = 0;
		re_bias = 1;
      end
8'd7: begin //weights conv2 1x1
		firstaddr = conv2_1;
		lastaddr = conv2_2;
		re_weights = 1;
		re_bias = 0;
	  end
8'd8: begin //bias conv2 1x1
		firstaddr = bias2_1;
		lastaddr = bias2_2;
		re_weights = 0;
		re_bias = 1;
	  end
8'd10: begin //weights conv3 dw 2
		firstaddr = conv2_2;
		lastaddr  = conv3_1;
		re_weights = 1;
		re_bias = 0;
	   end
8'd11: begin //bias conv3 DW
		firstaddr = bias2_2;
		lastaddr  = bias3_1;
		re_weights = 0;
		re_bias = 1;
	   end
8'd13: begin //weights conv3 1x1
   	firstaddr = conv3_1;
   	lastaddr  = conv3_2;
   	re_weights = 1;
   	re_bias = 0;
      end
8'd14: begin //bias conv
   	firstaddr = bias3_1;
   	lastaddr  = bias3_2;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd16: begin
   	firstaddr = conv3_2; // dw 3
   	lastaddr  = conv4_1;
   	re_weights = 1;
   	re_bias = 0;
      end
8'd17: begin
   	firstaddr = bias3_2;
   	lastaddr  = bias4_1;
   	re_weights = 0;
   	re_bias = 1;
     end
8'd19: begin
   	firstaddr = conv4_1;
   	lastaddr  = conv4_2;
   	re_weights = 1;
   	re_bias = 0;
      end
8'd20: begin
   	firstaddr = bias4_1;
   	lastaddr  = bias4_2;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd22: begin
   	firstaddr = conv4_2; // dw 4
   	lastaddr  = conv5_1;
   	re_weights = 1;
   	re_bias = 0;
     end
8'd23: begin
   	firstaddr = bias4_2;
   	lastaddr  = bias5_1;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd25: begin
   	firstaddr = conv5_1;
   	lastaddr  = conv5_2;
   	re_weights = 1;
   	re_bias = 0;
      end
8'd26: begin
   	firstaddr = bias5_1;
   	lastaddr  = bias5_2;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd28: begin
   	firstaddr = conv5_2; // dw 5
   	lastaddr  = conv6_1;
   	re_weights = 1;
   	re_bias = 0;
     end
8'd29: begin
   	firstaddr = bias5_2;
   	lastaddr  = bias6_1;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd31: begin
   	firstaddr = conv6_1;
   	lastaddr  = conv6_2;
   	re_weights = 1;
   	re_bias = 0;
      end
8'd32: begin
   	firstaddr = bias6_1;
   	lastaddr  = bias6_2;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd34: begin
   	firstaddr = conv6_2; // dw 6
   	lastaddr  = conv7_1;
   	re_weights = 1;
   	re_bias = 0;
      end
8'd35: begin
   	firstaddr = bias6_2;
   	lastaddr  = bias7_1;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd37: begin
   	firstaddr = conv7_1;
   	lastaddr  = conv7_2;
   	re_weights = 1;
   	re_bias = 0;
     end
8'd38: begin
   	firstaddr = bias7_1;
   	lastaddr  = bias7_2;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd40: begin
   	firstaddr = conv7_2; // dw 7
   	lastaddr  = conv8_1;
   	re_weights = 1;
   	re_bias = 0;
      end
8'd41: begin
   	firstaddr = bias7_2;
   	lastaddr  = bias8_1;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd43: begin
   	firstaddr = conv8_1;
   	lastaddr  = conv8_2;
   	re_weights = 1;
   	re_bias = 0;
      end
8'd44: begin
   	firstaddr = bias8_1;
   	lastaddr  = bias8_2;
   	re_weights = 0;
   	re_bias = 1;
      end
8'd46: begin
	    firstaddr = conv8_2; // dw 8
	    lastaddr  = conv9_1;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd47: begin
	    firstaddr = bias8_2;
	    lastaddr  = bias9_1;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd49: begin
	    firstaddr = conv9_1;
	    lastaddr  = conv9_2;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd50: begin
	    firstaddr = bias9_1;
	    lastaddr  = bias9_2;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd52: begin
	    firstaddr = conv9_2; // dw 9
	    lastaddr  = conv10_1;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd53: begin
	    firstaddr = bias9_2;
	    lastaddr  = bias10_1;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd55: begin
	    firstaddr = conv10_1;
	    lastaddr  = conv10_2;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd56: begin
	    firstaddr = bias10_1;
	    lastaddr  = bias10_2;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd58: begin
	    firstaddr = conv10_2; // dw 10
	    lastaddr  = conv11_1;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd59: begin
	    firstaddr = bias10_2;
	    lastaddr  = bias11_1;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd61: begin
	    firstaddr = conv11_1;
	    lastaddr  = conv11_2;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd62: begin
	    firstaddr = bias11_1;
	    lastaddr  = bias11_2;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd64: begin
	    firstaddr = conv11_2; // dw 11
	    lastaddr  = conv12_1;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd65: begin
	    firstaddr = bias11_2;
	    lastaddr  = bias12_1;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd67: begin
	    firstaddr = conv12_1;
	    lastaddr  = conv12_2;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd68: begin
	    firstaddr = bias12_1;
	    lastaddr  = bias12_2;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd70: begin
	    firstaddr = conv12_2; // dw 12
	    lastaddr  = conv13_1;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd71: begin
	    firstaddr = bias12_2;
	    lastaddr  = bias13_1;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd73: begin
	    firstaddr = conv13_1;
	    lastaddr  = conv13_2;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd74: begin
	    firstaddr = bias13_1;
	    lastaddr  = bias13_2;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd76: begin
	    firstaddr = conv13_2; // dw 13
	    lastaddr  = conv14_1;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd77: begin
	    firstaddr = bias13_2;
	    lastaddr  = bias14_1;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd79: begin
	    firstaddr = conv14_1;
	    lastaddr  = conv14_2_1;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd80: begin
	    firstaddr = bias14_1;
	    lastaddr  = bias14_2_1;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd82: begin
	    firstaddr = conv14_2_1;
	    lastaddr  = conv14_2_2;
	    re_weights = 1;
	    re_bias = 0;
       end
8'd83: begin
	    firstaddr = bias14_2_1;
	    lastaddr  = bias14_2_2;
	    re_weights = 0;
	    re_bias = 1;
       end
8'd85: begin
	    firstaddr = conv14_2_2;
	    lastaddr  = predict;
	    re_weights = 1;
	    re_bias = 0;
       end
default:
		begin
			re_weights = 0;
			re_bias = 0;
		end
endcase
endmodule
