module Seg7 (data,hex);

input [3:0] data;

output [6:0] hex;

assign hex[0] = !(((data==0)||(data==2)||(data==3)||(data==5)||(data==6)||(data==7)||(data==8)||(data==9)||(data==10)||(data==12)||(data==14)||(data==15))?1'b1:1'b0);
assign hex[1] = !(((data==0)||(data==1)||(data==2)||(data==3)||(data==4)||(data==7)||(data==8)||(data==9)||(data==10)||(data==13))?1'b1:1'b0);
assign hex[2] = !(((data==0)||(data==1)||(data==3)||(data==4)||(data==5)||(data==6)||(data==7)||(data==8)||(data==9)||(data==10)||(data==11)||(data==13))?1'b1:1'b0);
assign hex[3] = !(((data==0)||(data==2)||(data==3)||(data==5)||(data==6)||(data==8)||(data==9)||(data==11)||(data==12)||(data==13)||(data==14))?1'b1:1'b0);
assign hex[4] = !(((data==0)||(data==2)||(data==6)||(data==8)||(data==10)||(data==11)||(data==12)||(data==13)||(data==14)||(data==15))?1'b1:1'b0);
assign hex[5] = !(((data==0)||(data==4)||(data==5)||(data==6)||(data==8)||(data==9)||(data==10)||(data==11)||(data==12)||(data==14)||(data==15))?1'b1:1'b0);
assign hex[6] = !(((data==2)||(data==3)||(data==4)||(data==5)||(data==6)||(data==8)||(data==9)||(data==10)||(data==11)||(data==13)||(data==14)||(data==15))?1'b1:1'b0);

endmodule