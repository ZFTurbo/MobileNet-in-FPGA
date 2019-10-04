module dense(clk, dense_en, STOP, in, out, we, re_p, re_w, read_addressp, read_addressw, write_addressp, memstartp, memstartzap, qp, qw, res, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43, w51, w52, w53, w61, w62, w63, w71, w72, w73, w81, w82, w83, p11, p12, p13, p21, p22, p23, p31, p32, p33, p41, p42, p43, p51, p52, p53, p61, p62, p63, p71, p72, p73, p81, p82, p83, go, nozero);

parameter num_conv=0;

parameter SIZE_1=0;
parameter SIZE_2=0;
parameter SIZE_3=0;
parameter SIZE_4=0;
parameter SIZE_5=0;
parameter SIZE_6=0;
parameter SIZE_7=0;
parameter SIZE_8=0;
parameter SIZE_address_pix=0;
parameter SIZE_address_wei=0;
parameter SIZE_weights=0;

input clk,dense_en;
output reg STOP;
input [8:0] in;
input [1:0] out;
output reg we,re_p,re_w;
output reg [SIZE_address_pix-1:0] read_addressp;
output reg [SIZE_address_wei-1:0] read_addressw;
output reg [SIZE_address_pix-1:0] write_addressp;
input [SIZE_address_pix-1:0] memstartp,memstartzap;
input signed [SIZE_8-1:0] qp;
input signed [SIZE_weights*9-1:0] qw;
output reg signed [SIZE_8-1:0] res;
input signed [32-1:0] Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8;
output reg signed [SIZE_weights - 1:0] w11, w12, w13, w21, w22, w23, w31, w32, w33, w41, w42, w43, w51, w52, w53, w61, w62, w63, w71, w72, w73, w81, w82, w83;
output reg signed [SIZE_1-1:0] p11, p12, p13, p21, p22, p23, p31, p32, p33, p41, p42, p43, p51, p52, p53, p61, p62, p63, p71, p72, p73, p81, p82, p83;
output reg go;
input nozero;

reg signed[SIZE_weights - 1:0] w11_pre, w12_pre, w13_pre, w14_pre, w15_pre, w16_pre, w17_pre, w18_pre, w19_pre;
reg signed[SIZE_weights - 1:0] w21_pre, w22_pre, w23_pre, w24_pre, w25_pre, w26_pre, w27_pre, w28_pre, w29_pre;
reg signed[SIZE_weights - 1:0] w31_pre, w32_pre, w33_pre, w34_pre, w35_pre, w36_pre, w37_pre, w38_pre, w39_pre;
reg signed[SIZE_weights - 1:0] w41_pre, w42_pre, w43_pre, w44_pre, w45_pre, w46_pre, w47_pre, w48_pre, w49_pre;
reg signed[SIZE_weights - 1:0] w51_pre, w52_pre, w53_pre, w54_pre, w55_pre, w56_pre, w57_pre, w58_pre, w59_pre;
reg signed[SIZE_weights - 1:0] w61_pre, w62_pre, w63_pre, w64_pre, w65_pre, w66_pre, w67_pre, w68_pre, w69_pre;
reg signed[SIZE_weights - 1:0] w71_pre, w72_pre, w73_pre, w74_pre, w75_pre, w76_pre, w77_pre, w78_pre, w79_pre;
reg signed[SIZE_weights - 1:0] w81_pre, w82_pre, w83_pre, w84_pre, w85_pre, w86_pre, w87_pre, w88_pre, w89_pre;

reg signed[SIZE_1 - 1:0] p11_pre, p12_pre, p13_pre, p14_pre, p15_pre, p16_pre, p17_pre, p18_pre, p19_pre;
reg signed[SIZE_1 - 1:0] p21_pre, p22_pre, p23_pre, p24_pre, p25_pre, p26_pre, p27_pre, p28_pre, p29_pre;
reg signed[SIZE_1 - 1:0] p31_pre, p32_pre, p33_pre, p34_pre, p35_pre, p36_pre, p37_pre, p38_pre, p39_pre;
reg signed[SIZE_1 - 1:0] p41_pre, p42_pre, p43_pre, p44_pre, p45_pre, p46_pre, p47_pre, p48_pre, p49_pre;
reg signed[SIZE_1 - 1:0] p51_pre, p52_pre, p53_pre, p54_pre, p55_pre, p56_pre, p57_pre, p58_pre, p59_pre;
reg signed[SIZE_1 - 1:0] p61_pre, p62_pre, p63_pre, p64_pre, p65_pre, p66_pre, p67_pre, p68_pre, p69_pre;
reg signed[SIZE_1 - 1:0] p71_pre, p72_pre, p73_pre, p74_pre, p75_pre, p76_pre, p77_pre, p78_pre, p79_pre;
reg signed[SIZE_1 - 1:0] p81_pre, p82_pre, p83_pre, p84_pre, p85_pre, p86_pre, p87_pre, p88_pre, p89_pre;
reg [3:0] marker;
reg [6:0] lvl;
reg [8:0] i;
reg [8:0] j;
reg [2:0] sh;
reg signed [32-1:0] dp;
reg signed [SIZE_1-1:0] dp_shift;
reg signed [19-1:0]dp_check;

always @(posedge clk)
begin
    if (dense_en==1)
    begin
        re_p=1;
        case (marker)
2:begin
    if (i>(in>>3)+1) begin
        p11_pre = 0; p12_pre = 0; p13_pre = 0; p14_pre = 0; p15_pre = 0; p16_pre = 0; p17_pre = 0; p18_pre = 0;
    end
    else begin
        p11_pre = qp[SIZE_8 - 1:SIZE_7]; p12_pre = qp[SIZE_7 - 1:SIZE_6]; p13_pre = qp[SIZE_6 - 1:SIZE_5]; p14_pre = qp[SIZE_5 - 1:SIZE_4]; p15_pre = qp[SIZE_4 - 1:SIZE_3]; p16_pre = qp[SIZE_3 - 1:SIZE_2]; p17_pre = qp[SIZE_2 - 1:SIZE_1]; p18_pre = qp[SIZE_1 - 1:0];
    end
    go=0;
    w11_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; w12_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; w13_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; w14_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; w15_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; w16_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; w17_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; w18_pre=qw[SIZE_weights*2-1:SIZE_weights*1]; w19_pre=qw[SIZE_weights*1-1:SIZE_weights*0]; 
    read_addressw = lvl*29 + 2 + j*8;
    end
3:begin
    if (i>(in>>3)+1) begin
        p19_pre = 0; p21_pre = 0; p22_pre = 0; p23_pre = 0; p24_pre = 0; p25_pre = 0; p26_pre = 0; p27_pre = 0;
    end
    else begin
        p19_pre = qp[SIZE_8 - 1:SIZE_7]; p21_pre = qp[SIZE_7 - 1:SIZE_6]; p22_pre = qp[SIZE_6 - 1:SIZE_5]; p23_pre = qp[SIZE_5 - 1:SIZE_4]; p24_pre = qp[SIZE_4 - 1:SIZE_3]; p25_pre = qp[SIZE_3 - 1:SIZE_2]; p26_pre = qp[SIZE_2 - 1:SIZE_1]; p27_pre = qp[SIZE_1 - 1:0];
    end
    if (i!=3) dp=Y1+Y2+Y3+Y4+Y5+Y6+Y7+Y8+dp;
    w21_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; w22_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; w23_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; w24_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; w25_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; w26_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; w27_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; w28_pre=qw[SIZE_weights*2-1:SIZE_weights*1]; w29_pre=qw[SIZE_weights*1-1:SIZE_weights*0]; 
    read_addressw = lvl*29 + 3 + j*8;
    end
4:begin
    if (i>(in>>3)+1) begin
        p28_pre = 0; p29_pre = 0; p31_pre = 0; p32_pre = 0; p33_pre = 0; p34_pre = 0; p35_pre = 0; p36_pre = 0;
    end
    else begin
        p28_pre = qp[SIZE_8 - 1:SIZE_7]; p29_pre = qp[SIZE_7 - 1:SIZE_6]; p31_pre = qp[SIZE_6 - 1:SIZE_5]; p32_pre = qp[SIZE_5 - 1:SIZE_4]; p33_pre = qp[SIZE_4 - 1:SIZE_3]; p34_pre = qp[SIZE_3 - 1:SIZE_2]; p35_pre = qp[SIZE_2 - 1:SIZE_1]; p36_pre = qp[SIZE_1 - 1:0];
    end
    go=1;

    w31_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; w32_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; w33_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; w34_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; w35_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; w36_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; w37_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; w38_pre=qw[SIZE_weights*2-1:SIZE_weights*1]; w39_pre=qw[SIZE_weights*1-1:SIZE_weights*0]; 
    read_addressw = lvl*29 + 4 + j*8;
	p11=p11_pre; p12=p12_pre; p13=p13_pre; 
    p21=p14_pre; p22=p15_pre; p23=p16_pre; 
    p31=p17_pre; p32=p18_pre; p33=p19_pre; 
    p41=p21_pre; p42=p22_pre; p43=p23_pre; 
    p51=p24_pre; p52=p25_pre; p53=p26_pre; 
    p61=p27_pre; p62=p28_pre; p63=p29_pre; 
    p71=p31_pre; p72=p32_pre; p73=p33_pre; 
    p81=p34_pre; p82=p35_pre; p83=p36_pre; 

    w11=w11_pre; w12=w12_pre; w13=w13_pre; 
    w21=w14_pre; w22=w15_pre; w23=w16_pre; 
    w31=w17_pre; w32=w18_pre; w33=w19_pre; 
    w41=w21_pre; w42=w22_pre; w43=w23_pre; 
    w51=w24_pre; w52=w25_pre; w53=w26_pre; 
    w61=w27_pre; w62=w28_pre; w63=w29_pre; 
    w71=w31_pre; w72=w32_pre; w73=w33_pre; 
    w81=w34_pre; w82=w35_pre; w83=w36_pre; 
    end
5:begin
    if (i>(in>>3)+1) begin
        p37_pre = 0; p38_pre = 0; p39_pre = 0; p41_pre = 0; p42_pre = 0; p43_pre = 0; p44_pre = 0; p45_pre = 0;
    end
    else begin
        p37_pre = qp[SIZE_8 - 1:SIZE_7]; p38_pre = qp[SIZE_7 - 1:SIZE_6]; p39_pre = qp[SIZE_6 - 1:SIZE_5]; p41_pre = qp[SIZE_5 - 1:SIZE_4]; p42_pre = qp[SIZE_4 - 1:SIZE_3]; p43_pre = qp[SIZE_3 - 1:SIZE_2]; p44_pre = qp[SIZE_2 - 1:SIZE_1]; p45_pre = qp[SIZE_1 - 1:0];
    end
    go=0;
    w41_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; w42_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; w43_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; w44_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; w45_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; w46_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; w47_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; w48_pre=qw[SIZE_weights*2-1:SIZE_weights*1]; w49_pre=qw[SIZE_weights*1-1:SIZE_weights*0]; 
    read_addressw = lvl*29 + 5 + j*8;
    end
6:begin
    if (i>(in>>3)+1) begin
        p46_pre = 0; p47_pre = 0; p48_pre = 0; p49_pre = 0; p51_pre = 0; p52_pre = 0; p53_pre = 0; p54_pre = 0;
    end
    else begin
        p46_pre = qp[SIZE_8 - 1:SIZE_7]; p47_pre = qp[SIZE_7 - 1:SIZE_6]; p48_pre = qp[SIZE_6 - 1:SIZE_5]; p49_pre = qp[SIZE_5 - 1:SIZE_4]; p51_pre = qp[SIZE_4 - 1:SIZE_3]; p52_pre = qp[SIZE_3 - 1:SIZE_2]; p53_pre = qp[SIZE_2 - 1:SIZE_1]; p54_pre = qp[SIZE_1 - 1:0];
    end
    dp=Y1+Y2+Y3+Y4+Y5+Y6+Y7+Y8+dp;
    w51_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; w52_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; w53_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; w54_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; w55_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; w56_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; w57_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; w58_pre=qw[SIZE_weights*2-1:SIZE_weights*1]; w59_pre=qw[SIZE_weights*1-1:SIZE_weights*0]; 
    read_addressw = lvl*29 + 6 + j*8;
    end
7:begin
    if (i>(in>>3)+1) begin
        p55_pre = 0; p56_pre = 0; p57_pre = 0; p58_pre = 0; p59_pre = 0; p61_pre = 0; p62_pre = 0; p63_pre = 0;
    end
    else begin
        p55_pre = qp[SIZE_8 - 1:SIZE_7]; p56_pre = qp[SIZE_7 - 1:SIZE_6]; p57_pre = qp[SIZE_6 - 1:SIZE_5]; p58_pre = qp[SIZE_5 - 1:SIZE_4]; p59_pre = qp[SIZE_4 - 1:SIZE_3]; p61_pre = qp[SIZE_3 - 1:SIZE_2]; p62_pre = qp[SIZE_2 - 1:SIZE_1]; p63_pre = qp[SIZE_1 - 1:0];
    end
    go=1;
    w61_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; w62_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; w63_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; w64_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; w65_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; w66_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; w67_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; w68_pre=qw[SIZE_weights*2-1:SIZE_weights*1]; w69_pre=qw[SIZE_weights*1-1:SIZE_weights*0]; 
    read_addressw = lvl*29 + 7 + j*8;
	p11=p37_pre; p12=p38_pre; p13=p39_pre; 
    p21=p41_pre; p22=p42_pre; p23=p43_pre; 
    p31=p44_pre; p32=p45_pre; p33=p46_pre; 
    p41=p47_pre; p42=p48_pre; p43=p49_pre; 
    p51=p51_pre; p52=p52_pre; p53=p53_pre; 
    p61=p54_pre; p62=p55_pre; p63=p56_pre; 
    p71=p57_pre; p72=p58_pre; p73=p59_pre; 
    p81=p61_pre; p82=p62_pre; p83=p63_pre; 

    w11=w37_pre; w12=w38_pre; w13=w39_pre; 
    w21=w41_pre; w22=w42_pre; w23=w43_pre; 
    w31=w44_pre; w32=w45_pre; w33=w46_pre; 
    w41=w47_pre; w42=w48_pre; w43=w49_pre; 
    w51=w51_pre; w52=w52_pre; w53=w53_pre; 
    w61=w54_pre; w62=w55_pre; w63=w56_pre; 
    w71=w57_pre; w72=w58_pre; w73=w59_pre; 
    w81=w61_pre; w82=w62_pre; w83=w63_pre; 
    end
8:begin
    if (i>(in>>3)+1) begin
        p64_pre = 0; p65_pre = 0; p66_pre = 0; p67_pre = 0; p68_pre = 0; p69_pre = 0; p71_pre = 0; p72_pre = 0;
    end
    else begin
        p64_pre = qp[SIZE_8 - 1:SIZE_7]; p65_pre = qp[SIZE_7 - 1:SIZE_6]; p66_pre = qp[SIZE_6 - 1:SIZE_5]; p67_pre = qp[SIZE_5 - 1:SIZE_4]; p68_pre = qp[SIZE_4 - 1:SIZE_3]; p69_pre = qp[SIZE_3 - 1:SIZE_2]; p71_pre = qp[SIZE_2 - 1:SIZE_1]; p72_pre = qp[SIZE_1 - 1:0];
    end
    go=0;
    j=j+1;
    w71_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; w72_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; w73_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; w74_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; w75_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; w76_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; w77_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; w78_pre=qw[SIZE_weights*2-1:SIZE_weights*1]; w79_pre=qw[SIZE_weights*1-1:SIZE_weights*0]; 
    end
0:begin
    if (i>(in>>3)+1) begin
        p73_pre = 0; p74_pre = 0; p75_pre = 0; p76_pre = 0; p77_pre = 0; p78_pre = 0; p79_pre = 0; p81_pre = 0;
    end
    else begin
        p73_pre = qp[SIZE_8 - 1:SIZE_7]; p74_pre = qp[SIZE_7 - 1:SIZE_6]; p75_pre = qp[SIZE_6 - 1:SIZE_5]; p76_pre = qp[SIZE_5 - 1:SIZE_4]; p77_pre = qp[SIZE_4 - 1:SIZE_3]; p78_pre = qp[SIZE_3 - 1:SIZE_2]; p79_pre = qp[SIZE_2 - 1:SIZE_1]; p81_pre = qp[SIZE_1 - 1:0];
    end
    we=0;
    re_w=1;
    if (i!=0) dp=Y1+Y2+Y3+Y4+Y5+Y6+Y7+Y8+dp;
    w81_pre=qw[SIZE_weights*9-1:SIZE_weights*8]; w82_pre=qw[SIZE_weights*8-1:SIZE_weights*7]; w83_pre=qw[SIZE_weights*7-1:SIZE_weights*6]; w84_pre=qw[SIZE_weights*6-1:SIZE_weights*5]; w85_pre=qw[SIZE_weights*5-1:SIZE_weights*4]; w86_pre=qw[SIZE_weights*4-1:SIZE_weights*3]; w87_pre=qw[SIZE_weights*3-1:SIZE_weights*2]; w88_pre=qw[SIZE_weights*2-1:SIZE_weights*1]; w89_pre=qw[SIZE_weights*1-1:SIZE_weights*0]; 
    read_addressw = lvl*29 + 0 + j*8;
    end
1:begin
    if (i>(in>>3)+1) begin
        p82_pre = 0; p83_pre = 0; p84_pre = 0; p85_pre = 0; p86_pre = 0; p87_pre = 0; p88_pre = 0; p89_pre = 0;
    end
    else begin
        p82_pre = qp[SIZE_8 - 1:SIZE_7]; p83_pre = qp[SIZE_7 - 1:SIZE_6]; p84_pre = qp[SIZE_6 - 1:SIZE_5]; p85_pre = qp[SIZE_5 - 1:SIZE_4]; p86_pre = qp[SIZE_4 - 1:SIZE_3]; p87_pre = qp[SIZE_3 - 1:SIZE_2]; p88_pre = qp[SIZE_2 - 1:SIZE_1]; p89_pre = qp[SIZE_1 - 1:0];
    end
    if (i!=1) go=1;
    p11=p64_pre; p12=p65_pre; p13=p66_pre; 
    p21=p67_pre; p22=p68_pre; p23=p69_pre; 
    p31=p71_pre; p32=p72_pre; p33=p73_pre; 
    p41=p74_pre; p42=p75_pre; p43=p76_pre; 
    p51=p77_pre; p52=p78_pre; p53=p79_pre; 
    p61=p81_pre; p62=p82_pre; p63=p83_pre; 
    p71=p84_pre; p72=p85_pre; p73=p86_pre; 
    p81=p87_pre; p82=p88_pre; p83=p89_pre; 

    w11=w64_pre; w12=w65_pre; w13=w66_pre; 
    w21=w67_pre; w22=w68_pre; w23=w69_pre; 
    w31=w71_pre; w32=w72_pre; w33=w73_pre; 
    w41=w74_pre; w42=w75_pre; w43=w76_pre; 
    w51=w77_pre; w52=w78_pre; w53=w79_pre; 
    w61=w81_pre; w62=w82_pre; w63=w83_pre; 
    w71=w84_pre; w72=w85_pre; w73=w86_pre; 
    w81=w87_pre; w82=w88_pre; w83=w89_pre; 

    
    read_addressw = lvl*29 + 1 + j*8;
    end
            default: $display("Check case dense");
        endcase

        read_addressp=memstartp+i;

        if (marker!=8) marker=marker+1; else marker=0;
        i=i+1;
        if ((i>(in>>3)+4)&&(marker==4))
            begin
        	    write_addressp=memstartzap+(lvl>>(num_conv>>1));
                dp_check=dp[32-2:SIZE_1-2];
                if ((dp_shift<0)&&(nozero==0)) dp_shift=0;
		        if (dp_check>2**(SIZE_1-1)-1) dp_shift=2**(SIZE_1-1)-1;
                else dp_shift=dp_check;
                if (sh ==0) begin res=0; res[SIZE_8-1:SIZE_7]=dp_shift; end
                if (sh ==1) begin res[SIZE_7-1:SIZE_6]=dp_shift; end
                if (sh ==2) begin res[SIZE_6-1:SIZE_5]=dp_shift; end
                if (sh ==3) begin res[SIZE_5-1:SIZE_4]=dp_shift; end
                if (sh ==4) begin res[SIZE_4-1:SIZE_3]=dp_shift; end
                if (sh ==5) begin res[SIZE_3-1:SIZE_2]=dp_shift; end
                if (sh ==6) begin res[SIZE_2-1:SIZE_1]=dp_shift; end
                if (sh ==7) begin res[SIZE_1-1:0]=dp_shift; end
                lvl=lvl+1;
                i=0; 
                j=0; 
                dp=0; 
                marker=0;
                sh=sh+1; if (sh==num_conv) sh=0; 
		        if ((sh==0)||(lvl==out)) we=1;
                if (lvl==out) STOP=1;
    end
end
else
begin
    marker=0;
    i=0;
    j=0;
    sh=0;
    we=0;
    dp=0;
    res=0;
    re_p=0;
    re_w=0;
    STOP=0;
    lvl=0;
end
end
endmodule
