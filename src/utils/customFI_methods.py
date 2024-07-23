from pytorchfi.neuron_error_models import (
    random_batch_element,
    random_neuron_location,
)
from pytorchfi import core
import random
import logging


'''
this single_bit_flip_signed_across_batch_ours is our own custom bitflip function
'''
def single_bit_flip_signed_across_batch_ours(self, module, input_val, output):
        corrupt_conv_set = self.corrupt_layer
        range_max = self.get_conv_max(self.current_layer)
        logging.info(f"Current layer: {self.current_layer}")
        logging.info(f"Range_max: {range_max}")

        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                prev_value = output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info(f"Random Bit: {rand_bit}")
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = new_value

        else:
            if self.current_layer == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim[0]][
                    self.corrupt_dim[1]
                ][self.corrupt_dim[2]]

                rand_bit = random.randint(0, self.bits - 1)
                logging.info(f"Random Bit: {rand_bit}")
                new_value = self._flip_bit_signed(prev_value, range_max, rand_bit)

                output[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][
                    self.corrupt_dim[2]
                ] = new_value

        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()



'''
This is our custom fi function to target exact neuron 
'''
def random_neuron_single_bit_inj_ours(pfi: core.fault_injection, layer_ranges, layer, fi_c, fi_h, fi_w):
    # TODO Support multiple error models via list
    pfi.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi)
    (returned_layer, C, H, W) = random_neuron_location(pfi, layer)
    if returned_layer != layer:
        print("Problem: fi_layer is not equal to the returned layer")
    return pfi.declare_neuron_fi(
        batch=[batch],
        layer_num=[layer],
        #dim1=[C],
        #dim2=[H],
        #dim3=[W],
        dim1=[fi_c],
        dim2=[fi_h],
        dim3=[fi_w],
        function=pfi.single_bit_flip_signed_across_batch,
    )

'''
random_neuron_single_bit_inj_ours2
this one uses custombitflip function
'''
def random_neuron_single_bit_inj_ours2(pfi: core.fault_injection, layer_ranges, layer, fi_c, fi_h, fi_w):
    # TODO Support multiple error models via list
    pfi.set_conv_max(layer_ranges)

    batch = random_batch_element(pfi)
    (returned_layer, C, H, W) = random_neuron_location(pfi, layer)
    if returned_layer != layer:
        print("Problem: fi_layer is not equal to the returned layer")
    return pfi.declare_neuron_fi(
        batch=[batch],
        layer_num=[layer],
        #dim1=[C],
        #dim2=[H],
        #dim3=[W],
        dim1=[fi_c],
        dim2=[fi_h],
        dim3=[fi_w],
        function=pfi.single_bit_flip_signed_across_batch_ours,
    )
