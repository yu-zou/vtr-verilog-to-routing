module and_latch_yzou(
    clock,
	a_in,
	b_in,
	out
);

    // SIGNAL DECLARATIONS
    input	clock;
    input[2:0]	a_in;
    input[2:0]	b_in;

    output[2:0]	out;

    // ASSIGN STATEMENTS
    always @(posedge clock)
    begin
        out <= a_in & b_in;
    end

endmodule
