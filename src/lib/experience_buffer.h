/**
 * @file experience_buffer.h
 * @brief Thread-safe CSV experience buffer for active learning
 *
 * Stores (data_stats, config, actual_result) tuples from compression
 * calls to disk as CSV. Used by the retrain script to improve the
 * neural network model.
 */

#ifndef EXPERIENCE_BUFFER_H
#define EXPERIENCE_BUFFER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A single experience sample from a compression call.
 */
typedef struct {
    double entropy;
    double mad;
    double second_derivative;
    size_t data_size;
    double error_bound;
    int action;                 /* algo + quant*8 + shuffle*16 */
    double actual_ratio;
    double actual_comp_time_ms;
} ExperienceSample;

/**
 * Initialize the experience buffer.
 * Opens/creates the CSV file and writes header if new.
 *
 * @param csv_path Path to the CSV file
 * @return 0 on success, -1 on error
 */
int experience_buffer_init(const char* csv_path);

/**
 * Append a sample to the experience buffer.
 * Thread-safe (mutex-protected).
 *
 * @param sample Pointer to the sample to append
 * @return 0 on success, -1 on error
 */
int experience_buffer_append(const ExperienceSample* sample);

/**
 * Get the number of samples written this session.
 *
 * @return Number of samples appended since init
 */
size_t experience_buffer_count(void);

/**
 * Close the experience buffer file handle.
 */
void experience_buffer_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* EXPERIENCE_BUFFER_H */
