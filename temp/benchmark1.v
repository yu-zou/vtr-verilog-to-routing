module benchmark1(
    clock,
	a_in,
	b_in,
	out
);

    // SIGNAL DECLARATIONS
    input		clock;
    input[5:0]	a_in;
    input[5:0]	b_in;

    output		out;

	output[5:0] tempOut;
	assign tempOut[0] = a_in[0] ^ b_in[2];
	assign tempOut[1] = a_in[1] ^ b_in[2];
	assign tempOut[2] = a_in[2] | b_in[1] & b_in[0];
	assign tempOut[3] = ~a_in[3] | b_in[3] & a_in[4]
	assign tempOut[4] = b_in[4];

    // ASSIGN STATEMENTS
    always @(posedge clock)
    begin
        out <= tempOut[0] ^ tempOut[1] ^ tempOut[2] ^ tempOut[3] ^ tempOut[4];
    end

endmodule
