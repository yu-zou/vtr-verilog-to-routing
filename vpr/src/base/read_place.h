#ifndef READ_PLACE_H
#define READ_PLACE_H

void read_place(
    const char* net_file,
    const char* place_file,
    bool verify_file_hashes,
    const DeviceGrid& grid,
	/*Added by Yu Zou - 2019.11.3*/bool is_loading_initial_placement);

void print_place(const char* net_file,
                 const char* net_id,
                 const char* place_file);

void read_user_pad_loc(const char* pad_loc_file);

#endif
