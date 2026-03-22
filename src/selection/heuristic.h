#ifndef GPUCOMPRESS_HEURISTIC_H
#define GPUCOMPRESS_HEURISTIC_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Select a compression action based on entropy alone.
 *
 * Returns an action in the same encoding as the NN (algo_index 0-7,
 * no preprocessing).  This serves as a baseline to demonstrate
 * that the NN adds value beyond simple threshold rules.
 *
 * @param entropy  Byte-level Shannon entropy (0-8 bits)
 * @return Action index (0-31)
 */
int heuristic_select_action(double entropy);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_HEURISTIC_H */
