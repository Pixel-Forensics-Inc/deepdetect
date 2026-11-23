/**
 * DeepDetect
 * Copyright (c) 2026 Pixel Forensics, Inc.
 * Author: Cheni Chadowitz <cchadowitz@pixelforensics.com>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "onnxruntimeinputconns.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef USE_CUDA_CV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace dd
{
  void ImgONNXRuntimeInputFileConn::CVMatToORTBuffer(cv::Mat &img, int i)
  {
    cv::Mat converted;
    int channels = img.channels();

    if (_has_mean_scalar && _mean.size() != size_t(channels))
      throw InputConnectorBadParamException(
          "mean vector must match the number of channels ("
          + std::to_string(channels) + ")");

    if (!_std.empty() && _std.size() != size_t(channels))
      throw InputConnectorBadParamException(
          "std vector must match the number of channels ("
          + std::to_string(channels) + ")");

    img.convertTo(converted, CV_32F);
    size_t offset = size_t(i) * converted.total() * channels;
    for (int c = 0; c < channels; ++c)
      for (int h = 0; h < converted.rows; ++h)
        {
          const float *row = converted.ptr<float>(h);
          for (int w = 0; w < converted.cols; ++w)
            {
              float value = _scale * row[size_t(w) * channels + c];
              if (_has_mean_scalar)
                value -= _mean[c];
              if (!_std.empty())
                value /= _std[c];
              _buf[offset++] = value;
            }
        }
  }

#ifdef USE_CUDA_CV
  void ImgONNXRuntimeInputFileConn::GpuMatToORTBuffer(cv::cuda::GpuMat &img,
                                                      int i,
                                                      float *device_buffer)
  {
    int channels = img.channels();
    bool has_std = !_std.empty();
    cv::cuda::GpuMat converted;

    if (device_buffer == nullptr)
      throw InputConnectorInternalException(
          "No cuda buffer available to copy the data");

    if (_has_mean_scalar && _mean.size() != size_t(channels))
      throw InputConnectorBadParamException(
          "mean vector must match the number of channels ("
          + std::to_string(channels) + ")");

    if (!_std.empty() && _std.size() != size_t(channels))
      throw InputConnectorBadParamException(
          "std vector must match the number of channels ("
          + std::to_string(channels) + ")");

    img.convertTo(converted, CV_32F, 1.0, 0.0, *_cuda_stream);

    std::vector<cv::cuda::GpuMat> vec_channels;
    cv::cuda::split(converted, vec_channels, *_cuda_stream);

    for (int c = 0; c < channels; ++c)
      {
        auto &channel = vec_channels.at(c);
        cv::cuda::multiply(channel, _scale, channel, 1, -1, *_cuda_stream);

        if (_has_mean_scalar)
          cv::cuda::add(channel, -_mean[c], channel, cv::noArray(), -1,
                        *_cuda_stream);
        if (has_std)
          cv::cuda::multiply(channel, 1.0 / _std[c], channel, 1, -1,
                             *_cuda_stream);

        size_t offset
            = size_t(img.rows) * img.cols * (size_t(i) * channels + c);
        cudaMemcpy2DAsync(device_buffer + offset, img.cols * sizeof(float),
                          channel.ptr<float>(), channel.step,
                          img.cols * sizeof(float), img.rows,
                          cudaMemcpyDeviceToDevice,
                          cv::cuda::StreamAccessor::getStream(*_cuda_stream));
      }
  }
#endif

  void ImgONNXRuntimeInputFileConn::transform(
      oatpp::Object<DTO::ServicePredict> input_dto)
  {
    if (input_dto->has_mean_file)
      _logger->warn("mean file cannot be used with ONNX Runtime backend");

    try
      {
        ImgInputFileConn::transform(input_dto);
      }
    catch (const std::exception &e)
      {
        throw;
      }

    bool set_ids = this->_ids.empty();
    size_t img_count =
#ifdef USE_CUDA_CV
        _cuda ? this->_cuda_images.size() :
#endif
              this->_images.size();

    for (size_t i = 0; i < img_count; i++)
      {
        if (set_ids)
          this->_ids.push_back(this->_uris.at(i));
        _imgs_size.insert(std::pair<std::string, std::pair<int, int>>(
            this->_ids.at(i), this->_images_size.at(i)));
      }
    _batch_size = img_count;
    _batch_index = 0;
  }

  std::array<int, 3>
  ImgONNXRuntimeInputFileConn::batch_shape(int batch_size) const
  {
    auto shape_at = [&](int index) -> std::array<int, 3>
      {
#ifdef USE_CUDA_CV
        if (_cuda)
          {
            const auto &img = _cuda_images.at(index);
            return { img.channels(), img.rows, img.cols };
          }
#endif
        const auto &img = _images.at(index);
        return { img.channels(), img.rows, img.cols };
      };
    const auto shape = shape_at(_batch_index);
    if (shape[1] <= 0 || shape[2] <= 0)
      throw InputConnectorBadParamException("Empty transformed input image");
    for (int i = 1; i < batch_size; ++i)
      if (shape_at(_batch_index + i) != shape)
        throw InputConnectorBadParamException(
            "Transformed images in a batch must have identical channels, "
            "height and width");
    return shape;
  }

  void ImgONNXRuntimeInputFileConn::process_batch(int batch_size,
                                                  size_t elements,
                                                  float *device_buffer)
  {
#ifdef USE_CUDA_CV
    if (_cuda)
      {
        for (int i = 0; i < batch_size; ++i)
          GpuMatToORTBuffer(_cuda_images.at(_batch_index++), i, device_buffer);
        return;
      }
#else
    (void)device_buffer;
#endif
    _buf.resize(elements);
    for (int i = 0; i < batch_size; ++i)
      CVMatToORTBuffer(_images.at(_batch_index++), i);
  }
}
