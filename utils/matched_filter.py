import tensorflow as tf


def tf_weighted_inner(one, two, weight):
    """Compute weighted inner product of two frequency series, for the sigmasq rescaling.

    Keyword arguments:
        one -- batch array of templates in frequency domain
        two -- batch array of templates in frequency domain
        weight -- noise PSD in frequency domain
        
    Returns:
        out -- batch array of weighted inner products
    """
    return tf.math.reduce_sum(tf.math.conj(one) * two / weight, axis=1)


def tf_correlate(templates, samples):
    """Compute indexes of low and high frequency cutoffs.

    Keyword arguments:
        templates -- batch array of templates in frequency domain
        samples -- batch array of samples in frequency domain
        
    Returns:
        out -- batch array of correlated samples & templates
    """
    temp = tf.math.conj(templates)
    out = tf.math.multiply(temp, samples)
    return out


def tf_get_cutoff_indices(flow, fhigh, delta_f, N):
    """Compute indexes of low and high frequency cutoffs.

    Keyword arguments:
        flow -- low frequency cutoff
        fhigh -- high frequency cutoff
        delta_f -- frequency series sampling rate (default 1.0/2048)
        N -- 
        
    Returns:
        kmin -- index of low frequency cutoff
        kmax -- index of high frequency cutoff
    """
    if flow:
        kmin = int(flow / delta_f)
        if kmin < 0:
            err_msg = "Start frequency cannot be negative. "
            err_msg += "Supplied value and kmin {} and {}".format(flow, kmin)
            raise ValueError(err_msg)
    else:
        kmin = 1
    if fhigh:
        kmax = int(fhigh / delta_f)
        if kmax > int((N + 1)/2.):
            kmax = int((N + 1)/2.)
    else:
        # int() truncates towards 0, so this is
        # equivalent to the floor of the float
        kmax = int((N + 1)/2.)

    if kmax <= kmin:
        err_msg = "Kmax cannot be less than or equal to kmin. "
        err_msg += "Provided values of freqencies (min,max) were "
        err_msg += "{} and {} ".format(flow, fhigh)
        err_msg += "corresponding to (kmin, kmax) of "
        err_msg += "{} and {}.".format(kmin, kmax)
        raise ValueError(err_msg)

    return kmin,kmax


def tf_sigmasq(template, psd, N, kmin, kmax, delta_f):
    """Batch compute template normalization to rescale SNR time series.

    Keyword arguments:
        template -- batch array of templates in frequency domain
        psd -- array of PSD
        N -- 
        kmin -- index of low frequency cutoff
        kmax -- index of high frequency cutoff
        delta_f -- frequency series sampling rate (default 1.0/2048)
        
    Returns:
        template_norm -- batch array of normalization constants
    """
    norm = 4.0 * delta_f
    sq = tf_weighted_inner(template, template, psd)
    template_h_norm = tf.math.real(sq) * norm
    
    template_norm = tf.cast((4.0 * delta_f) / tf.math.sqrt(template_h_norm), dtype=tf.dtypes.complex128)
    template_norm = tf.reshape(template_norm, (template_norm.shape[0], 1))
    
    return template_norm


def matched_filter(sample, template, psd, N, kmin, kmax, duration, delta_t=1.0/2048, flow=30):
    """Batch compute SNR time series using matched filtering. Matched filtering is done row-wise for sample and template arrays.

    Keyword arguments:
        sample -- batch array of strain samples in frequency domain
        template -- batch array of templates in frequency domain
        psd -- array of PSD
        N -- 
        kmin -- index of low frequency cutoff
        kmax -- index of high frequency cutoff
        duration -- strain time series duration (seconds)
        delta_t -- strain time series sampling rate (default 1.0/2048)
        flow -- low frequency cutoff
        
    Returns:
        snr_ts -- batch array of SNR time series
    """
    with tf.device("/CPU:0"):
        delta_f = 1.0 / duration
        tsamples = int(duration / delta_t)
        flen = int(2048 / delta_f) + 1

        # Initialize _q output array
        _q = tf.zeros(N, dtype=sample.dtype)

        # Initialize SNR time series array
        sample_template_correlated_rescaled_padded_ts = tf.zeros(N, dtype=sample.dtype)

        # Correlate waveform with template
        sample_template_correlated = tf_correlate(template, sample)

        # Divide by PSD
        sample_template_correlated_rescaled = tf.math.divide(
            sample_template_correlated,
            psd
        )

        # Pad row to desired length
        paddings = tf.constant([[0,0], [kmin, _q.shape[0]-kmax]])
        sample_template_correlated_rescaled_padded = tf.pad(
            sample_template_correlated_rescaled, 
            paddings, 
            "CONSTANT"
        )

        # Convert to timeseries
        sample_template_correlated_rescaled_padded_ts = tf.signal.ifft(sample_template_correlated_rescaled_padded)
        sample_template_correlated_rescaled_padded_ts *= len(sample_template_correlated_rescaled_padded_ts[0])

        # Calculate normalization constant
        template_norm = tf_sigmasq(template, psd, N, kmin, kmax, delta_f)

        # Rescale SNR timeseries
        snr_ts = sample_template_correlated_rescaled_padded_ts * template_norm

        return snr_ts
    