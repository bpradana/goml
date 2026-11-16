//go:build darwin && cgo

#import "metal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface GMDevice : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property id<MTLCommandBuffer> current;
@property NSMutableArray *resources;
@property NSMutableArray *pending;
@property NSMutableDictionary *pipelines;
@property NSMutableDictionary *multiplications;
@property NSMutableDictionary *convolutions;
@property NSUInteger encoded;
@property uint64_t live;
@property uint64_t peak;
@property NSString *failure;
@end
@implementation GMDevice
@end

@interface GMBuffer : NSObject
@property id<MTLBuffer> buffer;
@property GMDevice *owner;
@property uint64_t bytes;
@end
@implementation GMBuffer

- (void)dealloc {
    _owner.live -= _bytes;
}

@end

static int fail(char **error, NSString *message) {
    if (error) {
        *error = strdup((message ?: @"unknown native failure").UTF8String);
    }
    return 0;
}

static void submit(GMDevice *d) {
    if (d.current) {
        [d.current commit];
        [d.pending addObject:@[ d.current, d.resources ]];
        d.current = nil;
        d.resources = nil;
        d.encoded = 0;
    }
}

static int reap(GMDevice *d, BOOL wait, char **error) {
    if (wait) {
        submit(d);
    }
    while (d.pending.count) {
        NSArray *entry = d.pending[0];
        id<MTLCommandBuffer> command = entry[0];
        if (wait) {
            [command waitUntilCompleted];
        }
        if (command.status < MTLCommandBufferStatusCompleted) {
            break;
        }
        if (command.error) {
            d.failure = command.error.localizedDescription;
        }
        [d.pending removeObjectAtIndex:0];
    }
    if (d.failure) {
        return fail(error, d.failure);
    }
    return 1;
}

static id<MTLCommandBuffer> command(GMDevice *d) {
    if (!d.current) {
        d.current = [d.queue commandBuffer];
        d.resources = [NSMutableArray array];
    }
    return d.current;
}

static void retainBuffers(GMDevice *d, GMBuffer *a, GMBuffer *b, GMBuffer *out) {
    if (a) {
        [d.resources addObject:a];
    }
    if (b) {
        [d.resources addObject:b];
    }
    if (out) {
        [d.resources addObject:out];
    }
}

void *gm_open(int index, const char *source, char **error) {
    @autoreleasepool {
        @try {
            GMDevice *d = [GMDevice new];
            if (index < 0) {
                d.device = MTLCreateSystemDefaultDevice();
            } else {
                NSArray *devices = MTLCopyAllDevices();
                if (index >= (int)devices.count) {
                    fail(error, @"device index outside available devices");
                    return NULL;
                }
                d.device = devices[index];
            }
            if (!d.device) {
                fail(error, @"no supported Metal device");
                return NULL;
            }
            d.queue = [d.device newCommandQueue];
            if (!d.queue) {
                fail(error, @"command queue allocation failed");
                return NULL;
            }
            MTLCompileOptions *options = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            NSError *nativeError = nil;
            id<MTLLibrary> library =
                [d.device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                       options:options
                                         error:&nativeError];
            if (!library) {
                fail(error, nativeError.localizedDescription);
                return NULL;
            }
            d.pipelines = [NSMutableDictionary dictionary];
            d.multiplications = [NSMutableDictionary dictionary];
            d.convolutions = [NSMutableDictionary dictionary];
            d.pending = [NSMutableArray array];
            for (NSString *name in library.functionNames) {
                id<MTLFunction> function = [library newFunctionWithName:name];
                id<MTLComputePipelineState> pipeline =
                    [d.device newComputePipelineStateWithFunction:function error:&nativeError];
                if (!pipeline) {
                    fail(error, nativeError.localizedDescription);
                    return NULL;
                }
                d.pipelines[name] = pipeline;
            }
            for (NSString *name in @[
                     @"fill_values", @"copy_values", @"binary_values", @"unary_values",
                     @"reduce_values"
                 ]) {
                if (!d.pipelines[name]) {
                    fail(error, [NSString stringWithFormat:@"missing shader %@", name]);
                    return NULL;
                }
            }
            return (__bridge_retained void *)d;
        } @catch (NSException *exception) {
            fail(error, exception.reason);
            return NULL;
        }
    }
}

const char *gm_name(void *device) {
    return ((__bridge GMDevice *)device).device.name.UTF8String;
}

void *gm_allocate(void *device, uint64_t bytes, char **error) {
    @autoreleasepool {
        @try {
            GMDevice *d = (__bridge GMDevice *)device;
            if (!reap(d, NO, error)) {
                return NULL;
            }
            uint64_t physical = MAX(bytes, 4);
            if (physical > d.device.maxBufferLength) {
                fail(error, @"allocation exceeds Metal maxBufferLength");
                return NULL;
            }
            GMBuffer *result = [GMBuffer new];
            result.buffer = [d.device newBufferWithLength:physical
                                                  options:MTLResourceStorageModeShared];
            if (!result.buffer) {
                fail(error, @"GPU allocation failed");
                return NULL;
            }
            result.owner = d;
            result.bytes = physical;
            d.live += physical;
            d.peak = MAX(d.peak, d.live);
            return (__bridge_retained void *)result;
        } @catch (NSException *exception) {
            fail(error, exception.reason);
            return NULL;
        }
    }
}

void gm_release_buffer(void *buffer) {
    @autoreleasepool {
        if (buffer) {
            GMBuffer *released = (__bridge_transfer GMBuffer *)buffer;
            (void)released;
        }
    }
}

int gm_sync(void *device, char **error) {
    @autoreleasepool {
        @try {
            return reap((__bridge GMDevice *)device, YES, error);
        } @catch (NSException *exception) {
            return fail(error, exception.reason);
        }
    }
}

int gm_upload(void *device, void *buffer, const float *data, uint64_t bytes, char **error) {
    @autoreleasepool {
        if (!gm_sync(device, error)) {
            return 0;
        }
        GMBuffer *b = (__bridge GMBuffer *)buffer;
        if (bytes) {
            memcpy(b.buffer.contents, data, bytes);
        }
        return 1;
    }
}

int gm_download(void *device, void *buffer, float *data, uint64_t bytes, char **error) {
    @autoreleasepool {
        if (!gm_sync(device, error)) {
            return 0;
        }
        GMBuffer *b = (__bridge GMBuffer *)buffer;
        if (bytes) {
            memcpy(data, b.buffer.contents, bytes);
        }
        return 1;
    }
}

void gm_close(void *device) {
    @autoreleasepool {
        GMDevice *d = (__bridge_transfer GMDevice *)device;
        reap(d, YES, NULL);
        d.convolutions = nil;
        d.multiplications = nil;
        d.pipelines = nil;
        d.queue = nil;
    }
}

void gm_stats(void *device, uint64_t *live, uint64_t *peak) {
    GMDevice *d = (__bridge GMDevice *)device;
    reap(d, NO, NULL);
    *live = d.live;
    *peak = d.peak;
}

int gm_compute(void *device, const char *name, void *a, void *b, void *out, GMParams *params,
               char **error) {
    @autoreleasepool {
        @try {
            GMDevice *d = (__bridge GMDevice *)device;
            if (!reap(d, NO, error)) {
                return 0;
            }
            if (params->n == 0) {
                return 1;
            }
            id<MTLComputePipelineState> pipeline =
                d.pipelines[[NSString stringWithUTF8String:name]];
            if (!pipeline) {
                return fail(error, @"shader unavailable");
            }
            GMBuffer *aa = (__bridge GMBuffer *)a, *bb = (__bridge GMBuffer *)b,
                     *oo = (__bridge GMBuffer *)out;
            id<MTLComputeCommandEncoder> encoder = [command(d) computeCommandEncoder];
            if (!encoder) {
                return fail(error, @"compute encoder allocation failed");
            }
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:aa.buffer offset:0 atIndex:0];
            [encoder setBuffer:bb.buffer offset:0 atIndex:1];
            [encoder setBuffer:oo.buffer offset:0 atIndex:2];
            [encoder setBytes:params length:sizeof(*params) atIndex:3];
            NSUInteger width = MIN(256, pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(params->n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            retainBuffers(d, aa, bb, oo);
            if (++d.encoded >= 64) {
                submit(d);
            }
            return 1;
        } @catch (NSException *exception) {
            return fail(error, exception.reason);
        }
    }
}

int gm_matmul(void *device, void *a, void *b, void *out, uint32_t m, uint32_t k, uint32_t n,
              uint32_t ao, uint32_t bo, uint32_t ar, uint32_t br, int ta, int tb, char **error) {
    @autoreleasepool {
        @try {
            GMDevice *d = (__bridge GMDevice *)device;
            if (!reap(d, NO, error)) {
                return 0;
            }
            GMBuffer *aa = (__bridge GMBuffer *)a, *bb = (__bridge GMBuffer *)b,
                     *oo = (__bridge GMBuffer *)out;
            if (ar < k || br < n || ((uint64_t)(m - 1) * ar + k + ao) * 4 > aa.buffer.length ||
                ((uint64_t)(k - 1) * br + n + bo) * 4 > bb.buffer.length ||
                (uint64_t)m * n * 4 > oo.buffer.length) {
                return fail(error, @"invalid matrix row stride or buffer bounds");
            }

            NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%d:%d", m, k, n, ta, tb];
            MPSMatrixMultiplication *multiplication = d.multiplications[key];
            if (!multiplication) {
                multiplication = [[MPSMatrixMultiplication alloc] initWithDevice:d.device
                                                                   transposeLeft:ta
                                                                  transposeRight:tb
                                                                      resultRows:m
                                                                   resultColumns:n
                                                                 interiorColumns:k
                                                                           alpha:1
                                                                            beta:0];
                d.multiplications[key] = multiplication;
            }
            MPSMatrixDescriptor *ad =
                [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                      columns:k
                                                     rowBytes:(NSUInteger)ar * 4
                                                     dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *bd =
                [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                      columns:n
                                                     rowBytes:(NSUInteger)br * 4
                                                     dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *od =
                [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                      columns:n
                                                     rowBytes:(NSUInteger)n * 4
                                                     dataType:MPSDataTypeFloat32];
            MPSMatrix *av = [[MPSMatrix alloc] initWithBuffer:aa.buffer
                                                       offset:(NSUInteger)ao * 4
                                                   descriptor:ad];
            MPSMatrix *bv = [[MPSMatrix alloc] initWithBuffer:bb.buffer
                                                       offset:(NSUInteger)bo * 4
                                                   descriptor:bd];
            MPSMatrix *ov = [[MPSMatrix alloc] initWithBuffer:oo.buffer descriptor:od];
            [multiplication encodeToCommandBuffer:command(d)
                                       leftMatrix:av
                                      rightMatrix:bv
                                     resultMatrix:ov];
            retainBuffers(d, aa, bb, oo);
            if (++d.encoded >= 64) {
                submit(d);
            }
            return 1;
        } @catch (NSException *exception) {
            return fail(error, exception.reason);
        }
    }
}

int gm_conv(void *device, void *input, void *weight, void *gradient, void *out, void *dw,
            GMParams *p, int backward, char **error) {
    @autoreleasepool {
        @try {
            GMDevice *d = (__bridge GMDevice *)device;
            if (!reap(d, NO, error)) {
                return 0;
            }
            submit(d);
            GMBuffer *xx = (__bridge GMBuffer *)input, *ww = (__bridge GMBuffer *)weight,
                     *gg = (__bridge GMBuffer *)gradient, *oo = (__bridge GMBuffer *)out,
                     *dww = (__bridge GMBuffer *)dw;
            NSUInteger n = p->shape[0], c = p->shape[1], h = p->shape[2], w = p->shape[3],
                       channels = p->ints[0], kernel = p->ints[1], stride = p->ints[2],
                       pad = p->ints[3];
            NSArray *xs = @[ @(n), @(c), @(h), @(w) ],
                    *ws = @[ @(c * kernel * kernel), @(channels) ];
            NSString *key =
                [NSString stringWithFormat:@"%@:%@:%lu:%lu:%lu:%d", xs, ws, (unsigned long)kernel,
                                           (unsigned long)stride, (unsigned long)pad, backward];
            NSArray *cached = d.convolutions[key];
            if (!cached) {
                MPSGraph *graph = [MPSGraph new];
                graph.options = MPSGraphOptionsNone;
                MPSGraphTensor *x = [graph placeholderWithShape:xs
                                                       dataType:MPSDataTypeFloat32
                                                           name:nil],
                               *weights = [graph placeholderWithShape:ws
                                                             dataType:MPSDataTypeFloat32
                                                                 name:nil];
                MPSGraphTensor *reshaped =
                    [graph reshapeTensor:weights
                               withShape:@[ @(c), @(kernel), @(kernel), @(channels) ]
                                    name:nil];
                MPSGraphTensor *oihw = [graph transposeTensor:reshaped
                                                  permutation:@[ @3, @0, @1, @2 ]
                                                         name:nil];
                MPSGraphConvolution2DOpDescriptor *descriptor = [MPSGraphConvolution2DOpDescriptor
                    descriptorWithStrideInX:stride
                                  strideInY:stride
                            dilationRateInX:1
                            dilationRateInY:1
                                     groups:1
                                paddingLeft:pad
                               paddingRight:pad
                                 paddingTop:pad
                              paddingBottom:pad
                               paddingStyle:MPSGraphPaddingStyleExplicit
                                 dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
                MPSGraphTensor *y = [graph convolution2DWithSourceTensor:x
                                                           weightsTensor:oihw
                                                              descriptor:descriptor
                                                                    name:nil];
                if (backward) {
                    MPSGraphTensor *g = [graph placeholderWithShape:y.shape
                                                           dataType:MPSDataTypeFloat32
                                                               name:nil];
                    MPSGraphTensor *dx =
                        [graph convolution2DDataGradientWithIncomingGradientTensor:g
                                                                     weightsTensor:oihw
                                                                       outputShape:xs
                                                      forwardConvolutionDescriptor:descriptor
                                                                              name:nil];
                    MPSGraphTensor *dweights =
                        [graph convolution2DWeightsGradientWithIncomingGradientTensor:g
                                                                         sourceTensor:x
                                                                          outputShape:oihw.shape
                                                         forwardConvolutionDescriptor:descriptor
                                                                                 name:nil];
                    MPSGraphTensor *ihwo = [graph transposeTensor:dweights
                                                      permutation:@[ @1, @2, @3, @0 ]
                                                             name:nil];
                    MPSGraphTensor *flat = [graph reshapeTensor:ihwo withShape:ws name:nil];
                    cached = @[ graph, x, weights, dx, g, flat ];
                } else {
                    cached = @[ graph, x, weights, y ];
                }
                d.convolutions[key] = cached;
            }
            MPSGraph *graph = cached[0];
            MPSGraphTensor *x = cached[1], *weights = cached[2], *y = cached[3];
            NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
            feeds[x] = [[MPSGraphTensorData alloc] initWithMTLBuffer:xx.buffer
                                                               shape:xs
                                                            dataType:MPSDataTypeFloat32];
            feeds[weights] = [[MPSGraphTensorData alloc] initWithMTLBuffer:ww.buffer
                                                                     shape:ws
                                                                  dataType:MPSDataTypeFloat32];
            NSMutableDictionary *results = [NSMutableDictionary dictionary];
            results[y] = [[MPSGraphTensorData alloc] initWithMTLBuffer:oo.buffer
                                                                 shape:y.shape
                                                              dataType:MPSDataTypeFloat32];
            if (backward) {
                MPSGraphTensor *g = cached[4], *dweight = cached[5];
                feeds[g] = [[MPSGraphTensorData alloc] initWithMTLBuffer:gg.buffer
                                                                   shape:g.shape
                                                                dataType:MPSDataTypeFloat32];
                results[dweight] =
                    [[MPSGraphTensorData alloc] initWithMTLBuffer:dww.buffer
                                                            shape:ws
                                                         dataType:MPSDataTypeFloat32];
            }
            MPSCommandBuffer *cb = [MPSCommandBuffer commandBufferFromCommandQueue:d.queue];
            MPSGraphExecutionDescriptor *execution = [MPSGraphExecutionDescriptor new];
            execution.waitUntilCompleted = NO;
            [graph encodeToCommandBuffer:cb
                                   feeds:feeds
                        targetOperations:nil
                       resultsDictionary:results
                     executionDescriptor:execution];
            id<MTLCommandBuffer> last = cb.commandBuffer;
            [cb commit];
            NSMutableArray *resources = [NSMutableArray arrayWithObjects:xx, ww, oo, nil];
            if (gg) {
                [resources addObject:gg];
            }
            if (dww) {
                [resources addObject:dww];
            }
            [d.pending addObject:@[ last, resources ]];
            return 1;
        } @catch (NSException *exception) {
            return fail(error, exception.reason);
        }
    }
}
