#ifdef _WIN32

#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <node_api.h>

#undef NAPI_EXTERN

static napi_status (*napi_adjust_external_memory_ptr)(napi_env, int64_t, int64_t*) = NULL;
static napi_status (*napi_call_function_ptr)(napi_env, napi_value, napi_value, size_t, const napi_value*, napi_value*) = NULL;
static napi_status (*napi_close_escapable_handle_scope_ptr)(napi_env, napi_escapable_handle_scope) = NULL;
static napi_status (*napi_close_handle_scope_ptr)(napi_env, napi_handle_scope) = NULL;
static napi_status (*napi_coerce_to_bool_ptr)(napi_env, napi_value, napi_value*) = NULL;
static napi_status (*napi_coerce_to_number_ptr)(napi_env, napi_value, napi_value*) = NULL;
static napi_status (*napi_coerce_to_object_ptr)(napi_env, napi_value, napi_value*) = NULL;
static napi_status (*napi_coerce_to_string_ptr)(napi_env, napi_value, napi_value*) = NULL;
static napi_status (*napi_create_array_ptr)(napi_env, napi_value*) = NULL;
static napi_status (*napi_create_array_with_length_ptr)(napi_env, size_t, napi_value*) = NULL;
static napi_status (*napi_create_arraybuffer_ptr)(napi_env, size_t, void**, napi_value*) = NULL;
static napi_status (*napi_create_dataview_ptr)(napi_env, size_t, napi_value, size_t, napi_value*) = NULL;
static napi_status (*napi_create_double_ptr)(napi_env, double, napi_value*) = NULL;
static napi_status (*napi_create_error_ptr)(napi_env, napi_value, napi_value, napi_value*) = NULL;
static napi_status (*napi_create_external_ptr)(napi_env, void*, napi_finalize, void*, napi_value*) = NULL;
static napi_status (*napi_create_external_arraybuffer_ptr)(napi_env, void*, size_t, napi_finalize, void*, napi_value*) = NULL;
static napi_status (*napi_create_function_ptr)(napi_env, const char*, size_t, napi_callback, void*, napi_value*) = NULL;
static napi_status (*napi_create_int32_ptr)(napi_env, int32_t, napi_value*) = NULL;
static napi_status (*napi_create_int64_ptr)(napi_env, int64_t, napi_value*) = NULL;
static napi_status (*napi_create_object_ptr)(napi_env, napi_value*) = NULL;
static napi_status (*napi_create_promise_ptr)(napi_env, napi_deferred*, napi_value*) = NULL;
static napi_status (*napi_create_range_error_ptr)(napi_env, napi_value, napi_value, napi_value*) = NULL;
static napi_status (*napi_create_reference_ptr)(napi_env, napi_value, uint32_t, napi_ref*) = NULL;
static napi_status (*napi_create_string_latin1_ptr)(napi_env, const char*, size_t, napi_value*) = NULL;
static napi_status (*napi_create_string_utf16_ptr)(napi_env, const char16_t*, size_t, napi_value*) = NULL;
static napi_status (*napi_create_string_utf8_ptr)(napi_env, const char*, size_t, napi_value*) = NULL;
static napi_status (*napi_create_symbol_ptr)(napi_env, napi_value, napi_value*) = NULL;
static napi_status (*napi_create_type_error_ptr)(napi_env, napi_value, napi_value, napi_value*) = NULL;
static napi_status (*napi_create_typedarray_ptr)(napi_env, napi_typedarray_type, size_t, napi_value, size_t, napi_value*) = NULL;
static napi_status (*napi_create_uint32_ptr)(napi_env, uint32_t, napi_value*) = NULL;
static napi_status (*napi_define_class_ptr)(napi_env, const char*, size_t, napi_callback, void*, size_t, const napi_property_descriptor*, napi_value*) = NULL;
static napi_status (*napi_define_properties_ptr)(napi_env, napi_value, size_t, const napi_property_descriptor*) = NULL;
static napi_status (*napi_delete_element_ptr)(napi_env, napi_value, uint32_t, bool*) = NULL;
static napi_status (*napi_delete_property_ptr)(napi_env, napi_value, napi_value, bool*) = NULL;
static napi_status (*napi_delete_reference_ptr)(napi_env, napi_ref) = NULL;
static napi_status (*napi_escape_handle_ptr)(napi_env, napi_escapable_handle_scope, napi_value, napi_value*) = NULL;
static napi_status (*napi_get_and_clear_last_exception_ptr)(napi_env, napi_value*) = NULL;
static napi_status (*napi_get_array_length_ptr)(napi_env, napi_value, uint32_t*) = NULL;
static napi_status (*napi_get_arraybuffer_info_ptr)(napi_env, napi_value, void**, size_t*) = NULL;
static napi_status (*napi_get_boolean_ptr)(napi_env, bool, napi_value*) = NULL;
static napi_status (*napi_get_cb_info_ptr)(napi_env, napi_callback_info, size_t*, napi_value*, napi_value*, void**) = NULL;
static napi_status (*napi_get_dataview_info_ptr)(napi_env, napi_value, size_t*, void**, napi_value*, size_t*) = NULL;
static napi_status (*napi_get_element_ptr)(napi_env, napi_value, uint32_t, napi_value*) = NULL;
static napi_status (*napi_get_global_ptr)(napi_env, napi_value*) = NULL;
static napi_status (*napi_get_last_error_info_ptr)(napi_env, const napi_extended_error_info**) = NULL;
static napi_status (*napi_get_named_property_ptr)(napi_env, napi_value, const char*, napi_value*) = NULL;
static napi_status (*napi_get_new_target_ptr)(napi_env, napi_callback_info, napi_value*) = NULL;
static napi_status (*napi_get_null_ptr)(napi_env, napi_value*) = NULL;
static napi_status (*napi_get_property_ptr)(napi_env, napi_value, napi_value, napi_value*) = NULL;
static napi_status (*napi_get_property_names_ptr)(napi_env, napi_value, napi_value*) = NULL;
static napi_status (*napi_get_prototype_ptr)(napi_env, napi_value, napi_value*) = NULL;
static napi_status (*napi_get_reference_value_ptr)(napi_env, napi_ref, napi_value*) = NULL;
static napi_status (*napi_get_typedarray_info_ptr)(napi_env, napi_value, napi_typedarray_type*, size_t*, void**, napi_value*, size_t*) = NULL;
static napi_status (*napi_get_undefined_ptr)(napi_env, napi_value*) = NULL;
static napi_status (*napi_get_value_bool_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_get_value_double_ptr)(napi_env, napi_value, double*) = NULL;
static napi_status (*napi_get_value_external_ptr)(napi_env, napi_value, void**) = NULL;
static napi_status (*napi_get_value_int32_ptr)(napi_env, napi_value, int32_t*) = NULL;
static napi_status (*napi_get_value_int64_ptr)(napi_env, napi_value, int64_t*) = NULL;
static napi_status (*napi_get_value_string_latin1_ptr)(napi_env, napi_value, char*, size_t, size_t*) = NULL;
static napi_status (*napi_get_value_string_utf16_ptr)(napi_env, napi_value, char16_t*, size_t, size_t*) = NULL;
static napi_status (*napi_get_value_string_utf8_ptr)(napi_env, napi_value, char*, size_t, size_t*) = NULL;
static napi_status (*napi_get_value_uint32_ptr)(napi_env, napi_value, uint32_t*) = NULL;
static napi_status (*napi_get_version_ptr)(napi_env, uint32_t*) = NULL;
static napi_status (*napi_has_element_ptr)(napi_env, napi_value, uint32_t, bool*) = NULL;
static napi_status (*napi_has_named_property_ptr)(napi_env, napi_value, const char*, bool*) = NULL;
static napi_status (*napi_has_own_property_ptr)(napi_env, napi_value, napi_value, bool*) = NULL;
static napi_status (*napi_has_property_ptr)(napi_env, napi_value, napi_value, bool*) = NULL;
static napi_status (*napi_instanceof_ptr)(napi_env, napi_value, napi_value, bool*) = NULL;
static napi_status (*napi_is_array_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_is_arraybuffer_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_is_dataview_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_is_error_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_is_exception_pending_ptr)(napi_env, bool*) = NULL;
static napi_status (*napi_is_promise_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_is_typedarray_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_new_instance_ptr)(napi_env, napi_value, size_t, const napi_value*, napi_value*) = NULL;
static napi_status (*napi_open_escapable_handle_scope_ptr)(napi_env, napi_escapable_handle_scope*) = NULL;
static napi_status (*napi_open_handle_scope_ptr)(napi_env, napi_handle_scope*) = NULL;
static napi_status (*napi_reference_ref_ptr)(napi_env, napi_ref, uint32_t*) = NULL;
static napi_status (*napi_reference_unref_ptr)(napi_env, napi_ref, uint32_t*) = NULL;
static napi_status (*napi_reject_deferred_ptr)(napi_env, napi_deferred, napi_value) = NULL;
static napi_status (*napi_remove_wrap_ptr)(napi_env, napi_value, void**) = NULL;
static napi_status (*napi_resolve_deferred_ptr)(napi_env, napi_deferred, napi_value) = NULL;
static napi_status (*napi_run_script_ptr)(napi_env, napi_value, napi_value*) = NULL;
static napi_status (*napi_set_element_ptr)(napi_env, napi_value, uint32_t, napi_value) = NULL;
static napi_status (*napi_set_named_property_ptr)(napi_env, napi_value, const char*, napi_value) = NULL;
static napi_status (*napi_set_property_ptr)(napi_env, napi_value, napi_value, napi_value) = NULL;
static napi_status (*napi_strict_equals_ptr)(napi_env, napi_value, napi_value, bool*) = NULL;
static napi_status (*napi_throw_ptr)(napi_env, napi_value) = NULL;
static napi_status (*napi_throw_error_ptr)(napi_env, const char*, const char*) = NULL;
static napi_status (*napi_throw_range_error_ptr)(napi_env, const char*, const char*) = NULL;
static napi_status (*napi_throw_type_error_ptr)(napi_env, const char*, const char*) = NULL;
static napi_status (*napi_typeof_ptr)(napi_env, napi_value, napi_valuetype*) = NULL;
static napi_status (*napi_unwrap_ptr)(napi_env, napi_value, void**) = NULL;
static napi_status (*napi_wrap_ptr)(napi_env, napi_value, void*, napi_finalize, void*, napi_ref*) = NULL;
static napi_status (*napi_async_destroy_ptr)(napi_env, napi_async_context) = NULL;
static napi_status (*napi_async_init_ptr)(napi_env, napi_value, napi_value, napi_async_context*) = NULL;
static napi_status (*napi_cancel_async_work_ptr)(napi_env, napi_async_work) = NULL;
static napi_status (*napi_create_async_work_ptr)(napi_env, napi_value, napi_value, napi_async_execute_callback, napi_async_complete_callback, void*, napi_async_work*) = NULL;
static napi_status (*napi_create_buffer_ptr)(napi_env, size_t, void**, napi_value*) = NULL;
static napi_status (*napi_create_buffer_copy_ptr)(napi_env, size_t, const void*, void**, napi_value*) = NULL;
static napi_status (*napi_create_external_buffer_ptr)(napi_env, size_t, void*, napi_finalize, void*, napi_value*) = NULL;
static napi_status (*napi_delete_async_work_ptr)(napi_env, napi_async_work) = NULL;
static void (*napi_fatal_error_ptr)(const char*, size_t, const char*, size_t) = NULL;
static napi_status (*napi_get_buffer_info_ptr)(napi_env, napi_value, void**, size_t*) = NULL;
static napi_status (*napi_get_node_version_ptr)(napi_env, const napi_node_version**) = NULL;
static napi_status (*napi_is_buffer_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_make_callback_ptr)(napi_env, napi_async_context, napi_value, napi_value, size_t, const napi_value*, napi_value*) = NULL;
static void (*napi_module_register_ptr)(napi_module*) = NULL;
static napi_status (*napi_queue_async_work_ptr)(napi_env, napi_async_work) = NULL;
static napi_status (*napi_get_uv_event_loop_ptr)(napi_env, struct uv_loop_s**) = NULL;
static napi_status (*napi_add_env_cleanup_hook_ptr)(napi_env, napi_cleanup_hook, void*) = NULL;
static napi_status (*napi_close_callback_scope_ptr)(napi_env, napi_callback_scope) = NULL;
static napi_status (*napi_fatal_exception_ptr)(napi_env, napi_value) = NULL;
static napi_status (*napi_open_callback_scope_ptr)(napi_env, napi_value, napi_async_context, napi_callback_scope*) = NULL;
static napi_status (*napi_remove_env_cleanup_hook_ptr)(napi_env, napi_cleanup_hook, void*) = NULL;
static napi_status (*napi_acquire_threadsafe_function_ptr)(napi_threadsafe_function) = NULL;
static napi_status (*napi_call_threadsafe_function_ptr)(napi_threadsafe_function, void*, napi_threadsafe_function_call_mode) = NULL;
static napi_status (*napi_create_threadsafe_function_ptr)(napi_env, napi_value, napi_value, napi_value, size_t, size_t, void*, napi_finalize, void*, napi_threadsafe_function_call_js, napi_threadsafe_function*) = NULL;
static napi_status (*napi_get_threadsafe_function_context_ptr)(napi_threadsafe_function, void**) = NULL;
static napi_status (*napi_ref_threadsafe_function_ptr)(napi_env, napi_threadsafe_function) = NULL;
static napi_status (*napi_release_threadsafe_function_ptr)(napi_threadsafe_function, napi_threadsafe_function_release_mode) = NULL;
static napi_status (*napi_unref_threadsafe_function_ptr)(napi_env, napi_threadsafe_function) = NULL;
static napi_status (*napi_add_finalizer_ptr)(napi_env, napi_value, void*, napi_finalize, void*, napi_ref*) = NULL;
static napi_status (*napi_create_date_ptr)(napi_env, double, napi_value*) = NULL;
static napi_status (*napi_get_date_value_ptr)(napi_env, napi_value, double*) = NULL;
static napi_status (*napi_is_date_ptr)(napi_env, napi_value, bool*) = NULL;
static napi_status (*napi_create_bigint_int64_ptr)(napi_env, int64_t, napi_value*) = NULL;
static napi_status (*napi_create_bigint_uint64_ptr)(napi_env, uint64_t, napi_value*) = NULL;
static napi_status (*napi_create_bigint_words_ptr)(napi_env, int, size_t, const uint64_t*, napi_value*) = NULL;
static napi_status (*napi_get_all_property_names_ptr)(napi_env, napi_value, napi_key_collection_mode, napi_key_filter, napi_key_conversion, napi_value*) = NULL;
static napi_status (*napi_get_instance_data_ptr)(napi_env, void**) = NULL;
static napi_status (*napi_get_value_bigint_int64_ptr)(napi_env, napi_value, int64_t*, bool*) = NULL;
static napi_status (*napi_get_value_bigint_uint64_ptr)(napi_env, napi_value, uint64_t*, bool*) = NULL;
static napi_status (*napi_get_value_bigint_words_ptr)(napi_env, napi_value, int*, size_t*, uint64_t*) = NULL;
static napi_status (*napi_set_instance_data_ptr)(napi_env, void*, napi_finalize, void*) = NULL;

static HMODULE napi_module_handle = NULL;
static int napi_functions_loaded = 0;

typedef struct {
    const char* name;
    void** func_ptr;
} napi_function_entry;

static napi_function_entry napi_functions[] = {
    {"napi_adjust_external_memory", (void**)&napi_adjust_external_memory_ptr},
    {"napi_call_function", (void**)&napi_call_function_ptr},
    {"napi_close_escapable_handle_scope", (void**)&napi_close_escapable_handle_scope_ptr},
    {"napi_close_handle_scope", (void**)&napi_close_handle_scope_ptr},
    {"napi_coerce_to_bool", (void**)&napi_coerce_to_bool_ptr},
    {"napi_coerce_to_number", (void**)&napi_coerce_to_number_ptr},
    {"napi_coerce_to_object", (void**)&napi_coerce_to_object_ptr},
    {"napi_coerce_to_string", (void**)&napi_coerce_to_string_ptr},
    {"napi_create_array", (void**)&napi_create_array_ptr},
    {"napi_create_array_with_length", (void**)&napi_create_array_with_length_ptr},
    {"napi_create_arraybuffer", (void**)&napi_create_arraybuffer_ptr},
    {"napi_create_dataview", (void**)&napi_create_dataview_ptr},
    {"napi_create_double", (void**)&napi_create_double_ptr},
    {"napi_create_error", (void**)&napi_create_error_ptr},
    {"napi_create_external", (void**)&napi_create_external_ptr},
    {"napi_create_external_arraybuffer", (void**)&napi_create_external_arraybuffer_ptr},
    {"napi_create_function", (void**)&napi_create_function_ptr},
    {"napi_create_int32", (void**)&napi_create_int32_ptr},
    {"napi_create_int64", (void**)&napi_create_int64_ptr},
    {"napi_create_object", (void**)&napi_create_object_ptr},
    {"napi_create_promise", (void**)&napi_create_promise_ptr},
    {"napi_create_range_error", (void**)&napi_create_range_error_ptr},
    {"napi_create_reference", (void**)&napi_create_reference_ptr},
    {"napi_create_string_latin1", (void**)&napi_create_string_latin1_ptr},
    {"napi_create_string_utf16", (void**)&napi_create_string_utf16_ptr},
    {"napi_create_string_utf8", (void**)&napi_create_string_utf8_ptr},
    {"napi_create_symbol", (void**)&napi_create_symbol_ptr},
    {"napi_create_type_error", (void**)&napi_create_type_error_ptr},
    {"napi_create_typedarray", (void**)&napi_create_typedarray_ptr},
    {"napi_create_uint32", (void**)&napi_create_uint32_ptr},
    {"napi_define_class", (void**)&napi_define_class_ptr},
    {"napi_define_properties", (void**)&napi_define_properties_ptr},
    {"napi_delete_element", (void**)&napi_delete_element_ptr},
    {"napi_delete_property", (void**)&napi_delete_property_ptr},
    {"napi_delete_reference", (void**)&napi_delete_reference_ptr},
    {"napi_escape_handle", (void**)&napi_escape_handle_ptr},
    {"napi_get_and_clear_last_exception", (void**)&napi_get_and_clear_last_exception_ptr},
    {"napi_get_array_length", (void**)&napi_get_array_length_ptr},
    {"napi_get_arraybuffer_info", (void**)&napi_get_arraybuffer_info_ptr},
    {"napi_get_boolean", (void**)&napi_get_boolean_ptr},
    {"napi_get_cb_info", (void**)&napi_get_cb_info_ptr},
    {"napi_get_dataview_info", (void**)&napi_get_dataview_info_ptr},
    {"napi_get_element", (void**)&napi_get_element_ptr},
    {"napi_get_global", (void**)&napi_get_global_ptr},
    {"napi_get_last_error_info", (void**)&napi_get_last_error_info_ptr},
    {"napi_get_named_property", (void**)&napi_get_named_property_ptr},
    {"napi_get_new_target", (void**)&napi_get_new_target_ptr},
    {"napi_get_null", (void**)&napi_get_null_ptr},
    {"napi_get_property", (void**)&napi_get_property_ptr},
    {"napi_get_property_names", (void**)&napi_get_property_names_ptr},
    {"napi_get_prototype", (void**)&napi_get_prototype_ptr},
    {"napi_get_reference_value", (void**)&napi_get_reference_value_ptr},
    {"napi_get_typedarray_info", (void**)&napi_get_typedarray_info_ptr},
    {"napi_get_undefined", (void**)&napi_get_undefined_ptr},
    {"napi_get_value_bool", (void**)&napi_get_value_bool_ptr},
    {"napi_get_value_double", (void**)&napi_get_value_double_ptr},
    {"napi_get_value_external", (void**)&napi_get_value_external_ptr},
    {"napi_get_value_int32", (void**)&napi_get_value_int32_ptr},
    {"napi_get_value_int64", (void**)&napi_get_value_int64_ptr},
    {"napi_get_value_string_latin1", (void**)&napi_get_value_string_latin1_ptr},
    {"napi_get_value_string_utf16", (void**)&napi_get_value_string_utf16_ptr},
    {"napi_get_value_string_utf8", (void**)&napi_get_value_string_utf8_ptr},
    {"napi_get_value_uint32", (void**)&napi_get_value_uint32_ptr},
    {"napi_get_version", (void**)&napi_get_version_ptr},
    {"napi_has_element", (void**)&napi_has_element_ptr},
    {"napi_has_named_property", (void**)&napi_has_named_property_ptr},
    {"napi_has_own_property", (void**)&napi_has_own_property_ptr},
    {"napi_has_property", (void**)&napi_has_property_ptr},
    {"napi_instanceof", (void**)&napi_instanceof_ptr},
    {"napi_is_array", (void**)&napi_is_array_ptr},
    {"napi_is_arraybuffer", (void**)&napi_is_arraybuffer_ptr},
    {"napi_is_dataview", (void**)&napi_is_dataview_ptr},
    {"napi_is_error", (void**)&napi_is_error_ptr},
    {"napi_is_exception_pending", (void**)&napi_is_exception_pending_ptr},
    {"napi_is_promise", (void**)&napi_is_promise_ptr},
    {"napi_is_typedarray", (void**)&napi_is_typedarray_ptr},
    {"napi_new_instance", (void**)&napi_new_instance_ptr},
    {"napi_open_escapable_handle_scope", (void**)&napi_open_escapable_handle_scope_ptr},
    {"napi_open_handle_scope", (void**)&napi_open_handle_scope_ptr},
    {"napi_reference_ref", (void**)&napi_reference_ref_ptr},
    {"napi_reference_unref", (void**)&napi_reference_unref_ptr},
    {"napi_reject_deferred", (void**)&napi_reject_deferred_ptr},
    {"napi_remove_wrap", (void**)&napi_remove_wrap_ptr},
    {"napi_resolve_deferred", (void**)&napi_resolve_deferred_ptr},
    {"napi_run_script", (void**)&napi_run_script_ptr},
    {"napi_set_element", (void**)&napi_set_element_ptr},
    {"napi_set_named_property", (void**)&napi_set_named_property_ptr},
    {"napi_set_property", (void**)&napi_set_property_ptr},
    {"napi_strict_equals", (void**)&napi_strict_equals_ptr},
    {"napi_throw", (void**)&napi_throw_ptr},
    {"napi_throw_error", (void**)&napi_throw_error_ptr},
    {"napi_throw_range_error", (void**)&napi_throw_range_error_ptr},
    {"napi_throw_type_error", (void**)&napi_throw_type_error_ptr},
    {"napi_typeof", (void**)&napi_typeof_ptr},
    {"napi_unwrap", (void**)&napi_unwrap_ptr},
    {"napi_wrap", (void**)&napi_wrap_ptr},
    {"napi_async_destroy", (void**)&napi_async_destroy_ptr},
    {"napi_async_init", (void**)&napi_async_init_ptr},
    {"napi_cancel_async_work", (void**)&napi_cancel_async_work_ptr},
    {"napi_create_async_work", (void**)&napi_create_async_work_ptr},
    {"napi_create_buffer", (void**)&napi_create_buffer_ptr},
    {"napi_create_buffer_copy", (void**)&napi_create_buffer_copy_ptr},
    {"napi_create_external_buffer", (void**)&napi_create_external_buffer_ptr},
    {"napi_delete_async_work", (void**)&napi_delete_async_work_ptr},
    {"napi_fatal_error", (void**)&napi_fatal_error_ptr},
    {"napi_get_buffer_info", (void**)&napi_get_buffer_info_ptr},
    {"napi_get_node_version", (void**)&napi_get_node_version_ptr},
    {"napi_is_buffer", (void**)&napi_is_buffer_ptr},
    {"napi_make_callback", (void**)&napi_make_callback_ptr},
    {"napi_module_register", (void**)&napi_module_register_ptr},
    {"napi_queue_async_work", (void**)&napi_queue_async_work_ptr},
    {"napi_get_uv_event_loop", (void**)&napi_get_uv_event_loop_ptr},
    {"napi_add_env_cleanup_hook", (void**)&napi_add_env_cleanup_hook_ptr},
    {"napi_close_callback_scope", (void**)&napi_close_callback_scope_ptr},
    {"napi_fatal_exception", (void**)&napi_fatal_exception_ptr},
    {"napi_open_callback_scope", (void**)&napi_open_callback_scope_ptr},
    {"napi_remove_env_cleanup_hook", (void**)&napi_remove_env_cleanup_hook_ptr},
    {"napi_acquire_threadsafe_function", (void**)&napi_acquire_threadsafe_function_ptr},
    {"napi_call_threadsafe_function", (void**)&napi_call_threadsafe_function_ptr},
    {"napi_create_threadsafe_function", (void**)&napi_create_threadsafe_function_ptr},
    {"napi_get_threadsafe_function_context", (void**)&napi_get_threadsafe_function_context_ptr},
    {"napi_ref_threadsafe_function", (void**)&napi_ref_threadsafe_function_ptr},
    {"napi_release_threadsafe_function", (void**)&napi_release_threadsafe_function_ptr},
    {"napi_unref_threadsafe_function", (void**)&napi_unref_threadsafe_function_ptr},
    {"napi_add_finalizer", (void**)&napi_add_finalizer_ptr},
    {"napi_create_date", (void**)&napi_create_date_ptr},
    {"napi_get_date_value", (void**)&napi_get_date_value_ptr},
    {"napi_is_date", (void**)&napi_is_date_ptr},
    {"napi_create_bigint_int64", (void**)&napi_create_bigint_int64_ptr},
    {"napi_create_bigint_uint64", (void**)&napi_create_bigint_uint64_ptr},
    {"napi_create_bigint_words", (void**)&napi_create_bigint_words_ptr},
    {"napi_get_all_property_names", (void**)&napi_get_all_property_names_ptr},
    {"napi_get_instance_data", (void**)&napi_get_instance_data_ptr},
    {"napi_get_value_bigint_int64", (void**)&napi_get_value_bigint_int64_ptr},
    {"napi_get_value_bigint_uint64", (void**)&napi_get_value_bigint_uint64_ptr},
    {"napi_get_value_bigint_words", (void**)&napi_get_value_bigint_words_ptr},
    {"napi_set_instance_data", (void**)&napi_set_instance_data_ptr},
    {NULL, NULL} // END MARK
};

static int load_napi_functions_from_module(HMODULE module) {
    if (!module) return 0;
    
    int loaded_count = 0;
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        FARPROC proc = GetProcAddress(module, napi_functions[i].name);
        if (proc) {
            *(napi_functions[i].func_ptr) = (void*)proc;
            loaded_count++;
        }
    }
    
    return loaded_count;
}

static int discover_and_load_napi_functions() {
    if (napi_functions_loaded) {
        return 1;
    }
    
    const char* module_names[] = {
        "node.exe",
        "node.dll",
        "electron.exe",
        "nw.exe",
        NULL
    };
    
    HMODULE current_process = GetModuleHandle(NULL);
    if (load_napi_functions_from_module(current_process) > 0) {
        napi_module_handle = current_process;
        napi_functions_loaded = 1;
        return 1;
    }
    
    for (int i = 0; module_names[i] != NULL; i++) {
        HMODULE module = GetModuleHandle(module_names[i]);
        if (module && load_napi_functions_from_module(module) > 0) {
            napi_module_handle = module;
            napi_functions_loaded = 1;
            return 1;
        }
    }
    
    HMODULE node_dll = LoadLibrary("node.dll");
    if (node_dll && load_napi_functions_from_module(node_dll) > 0) {
        napi_module_handle = node_dll;
        napi_functions_loaded = 1;
        return 1;
    }
    
    return 0;
}


napi_status napi_adjust_external_memory(napi_env env, int64_t change_in_bytes, int64_t* adjusted_value) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_adjust_external_memory_ptr) {
        return napi_generic_failure;
    }
    
    return napi_adjust_external_memory_ptr(env, change_in_bytes, adjusted_value);
}

napi_status napi_call_function(napi_env env, napi_value recv, napi_value func, size_t argc, const napi_value* argv, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_call_function_ptr) {
        return napi_generic_failure;
    }
    
    return napi_call_function_ptr(env, recv, func, argc, argv, result);
}

napi_status napi_close_escapable_handle_scope(napi_env env, napi_escapable_handle_scope scope) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_close_escapable_handle_scope_ptr) {
        return napi_generic_failure;
    }
    
    return napi_close_escapable_handle_scope_ptr(env, scope);
}

napi_status napi_close_handle_scope(napi_env env, napi_handle_scope scope) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_close_handle_scope_ptr) {
        return napi_generic_failure;
    }
    
    return napi_close_handle_scope_ptr(env, scope);
}

napi_status napi_coerce_to_bool(napi_env env, napi_value value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_coerce_to_bool_ptr) {
        return napi_generic_failure;
    }
    
    return napi_coerce_to_bool_ptr(env, value, result);
}

napi_status napi_coerce_to_number(napi_env env, napi_value value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_coerce_to_number_ptr) {
        return napi_generic_failure;
    }
    
    return napi_coerce_to_number_ptr(env, value, result);
}

napi_status napi_coerce_to_object(napi_env env, napi_value value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_coerce_to_object_ptr) {
        return napi_generic_failure;
    }
    
    return napi_coerce_to_object_ptr(env, value, result);
}

napi_status napi_coerce_to_string(napi_env env, napi_value value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_coerce_to_string_ptr) {
        return napi_generic_failure;
    }
    
    return napi_coerce_to_string_ptr(env, value, result);
}

napi_status napi_create_array(napi_env env, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_array_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_array_ptr(env, result);
}

napi_status napi_create_array_with_length(napi_env env, size_t length, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_array_with_length_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_array_with_length_ptr(env, length, result);
}

napi_status napi_create_arraybuffer(napi_env env, size_t byte_length, void** data, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_arraybuffer_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_arraybuffer_ptr(env, byte_length, data, result);
}

napi_status napi_create_dataview(napi_env env, size_t length, napi_value arraybuffer, size_t byte_offset, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_dataview_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_dataview_ptr(env, length, arraybuffer, byte_offset, result);
}

napi_status napi_create_double(napi_env env, double value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_double_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_double_ptr(env, value, result);
}

napi_status napi_create_error(napi_env env, napi_value code, napi_value msg, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_error_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_error_ptr(env, code, msg, result);
}

napi_status napi_create_external(napi_env env, void* data, napi_finalize finalize_cb, void* finalize_hint, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_external_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_external_ptr(env, data, finalize_cb, finalize_hint, result);
}

napi_status napi_create_external_arraybuffer(napi_env env, void* external_data, size_t byte_length, napi_finalize finalize_cb, void* finalize_hint, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_external_arraybuffer_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_external_arraybuffer_ptr(env, external_data, byte_length, finalize_cb, finalize_hint, result);
}

napi_status napi_create_function(napi_env env, const char* utf8name, size_t length, napi_callback cb, void* data, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_function_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_function_ptr(env, utf8name, length, cb, data, result);
}

napi_status napi_create_int32(napi_env env, int32_t value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_int32_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_int32_ptr(env, value, result);
}

napi_status napi_create_int64(napi_env env, int64_t value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_int64_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_int64_ptr(env, value, result);
}

napi_status napi_create_object(napi_env env, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_object_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_object_ptr(env, result);
}

napi_status napi_create_promise(napi_env env, napi_deferred* deferred, napi_value* promise) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_promise_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_promise_ptr(env, deferred, promise);
}

napi_status napi_create_range_error(napi_env env, napi_value code, napi_value msg, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_range_error_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_range_error_ptr(env, code, msg, result);
}

napi_status napi_create_reference(napi_env env, napi_value value, uint32_t initial_refcount, napi_ref* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_reference_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_reference_ptr(env, value, initial_refcount, result);
}

napi_status napi_create_string_latin1(napi_env env, const char* str, size_t length, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_string_latin1_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_string_latin1_ptr(env, str, length, result);
}

napi_status napi_create_string_utf16(napi_env env, const char16_t* str, size_t length, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_string_utf16_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_string_utf16_ptr(env, str, length, result);
}

napi_status napi_create_string_utf8(napi_env env, const char* str, size_t length, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_string_utf8_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_string_utf8_ptr(env, str, length, result);
}

napi_status napi_create_symbol(napi_env env, napi_value description, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_symbol_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_symbol_ptr(env, description, result);
}

napi_status napi_create_type_error(napi_env env, napi_value code, napi_value msg, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_type_error_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_type_error_ptr(env, code, msg, result);
}

napi_status napi_create_typedarray(napi_env env, napi_typedarray_type type, size_t length, napi_value arraybuffer, size_t byte_offset, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_typedarray_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_typedarray_ptr(env, type, length, arraybuffer, byte_offset, result);
}

napi_status napi_create_uint32(napi_env env, uint32_t value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_uint32_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_uint32_ptr(env, value, result);
}

napi_status napi_define_class(napi_env env, const char* utf8name, size_t length, napi_callback constructor, void* data, size_t property_count, const napi_property_descriptor* properties, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_define_class_ptr) {
        return napi_generic_failure;
    }
    
    return napi_define_class_ptr(env, utf8name, length, constructor, data, property_count, properties, result);
}

napi_status napi_define_properties(napi_env env, napi_value object, size_t property_count, const napi_property_descriptor* properties) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_define_properties_ptr) {
        return napi_generic_failure;
    }
    
    return napi_define_properties_ptr(env, object, property_count, properties);
}

napi_status napi_delete_element(napi_env env, napi_value object, uint32_t index, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_delete_element_ptr) {
        return napi_generic_failure;
    }
    
    return napi_delete_element_ptr(env, object, index, result);
}

napi_status napi_delete_property(napi_env env, napi_value object, napi_value key, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_delete_property_ptr) {
        return napi_generic_failure;
    }
    
    return napi_delete_property_ptr(env, object, key, result);
}

napi_status napi_delete_reference(napi_env env, napi_ref ref) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_delete_reference_ptr) {
        return napi_generic_failure;
    }
    
    return napi_delete_reference_ptr(env, ref);
}

napi_status napi_escape_handle(napi_env env, napi_escapable_handle_scope scope, napi_value escapee, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_escape_handle_ptr) {
        return napi_generic_failure;
    }
    
    return napi_escape_handle_ptr(env, scope, escapee, result);
}

napi_status napi_get_and_clear_last_exception(napi_env env, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_and_clear_last_exception_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_and_clear_last_exception_ptr(env, result);
}

napi_status napi_get_array_length(napi_env env, napi_value value, uint32_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_array_length_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_array_length_ptr(env, value, result);
}

napi_status napi_get_arraybuffer_info(napi_env env, napi_value arraybuffer, void** data, size_t* byte_length) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_arraybuffer_info_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_arraybuffer_info_ptr(env, arraybuffer, data, byte_length);
}

napi_status napi_get_boolean(napi_env env, bool value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_boolean_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_boolean_ptr(env, value, result);
}

napi_status napi_get_cb_info(napi_env env, napi_callback_info cbinfo, size_t* argc, napi_value* argv, napi_value* this_arg, void** data) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_cb_info_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_cb_info_ptr(env, cbinfo, argc, argv, this_arg, data);
}

napi_status napi_get_dataview_info(napi_env env, napi_value dataview, size_t* bytelength, void** data, napi_value* arraybuffer, size_t* byte_offset) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_dataview_info_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_dataview_info_ptr(env, dataview, bytelength, data, arraybuffer, byte_offset);
}

napi_status napi_get_element(napi_env env, napi_value object, uint32_t index, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_element_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_element_ptr(env, object, index, result);
}

napi_status napi_get_global(napi_env env, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_global_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_global_ptr(env, result);
}

napi_status napi_get_last_error_info(napi_env env, const napi_extended_error_info** result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_last_error_info_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_last_error_info_ptr(env, result);
}

napi_status napi_get_named_property(napi_env env, napi_value object, const char* utf8name, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_named_property_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_named_property_ptr(env, object, utf8name, result);
}

napi_status napi_get_new_target(napi_env env, napi_callback_info cbinfo, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_new_target_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_new_target_ptr(env, cbinfo, result);
}

napi_status napi_get_null(napi_env env, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_null_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_null_ptr(env, result);
}

napi_status napi_get_property(napi_env env, napi_value object, napi_value key, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_property_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_property_ptr(env, object, key, result);
}

napi_status napi_get_property_names(napi_env env, napi_value object, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_property_names_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_property_names_ptr(env, object, result);
}

napi_status napi_get_prototype(napi_env env, napi_value object, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_prototype_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_prototype_ptr(env, object, result);
}

napi_status napi_get_reference_value(napi_env env, napi_ref ref, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_reference_value_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_reference_value_ptr(env, ref, result);
}

napi_status napi_get_typedarray_info(napi_env env, napi_value typedarray, napi_typedarray_type* type, size_t* length, void** data, napi_value* arraybuffer, size_t* byte_offset) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_typedarray_info_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_typedarray_info_ptr(env, typedarray, type, length, data, arraybuffer, byte_offset);
}

napi_status napi_get_undefined(napi_env env, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_undefined_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_undefined_ptr(env, result);
}

napi_status napi_get_value_bool(napi_env env, napi_value value, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_bool_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_bool_ptr(env, value, result);
}

napi_status napi_get_value_double(napi_env env, napi_value value, double* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_double_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_double_ptr(env, value, result);
}

napi_status napi_get_value_external(napi_env env, napi_value value, void** result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_external_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_external_ptr(env, value, result);
}

napi_status napi_get_value_int32(napi_env env, napi_value value, int32_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_int32_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_int32_ptr(env, value, result);
}

napi_status napi_get_value_int64(napi_env env, napi_value value, int64_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_int64_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_int64_ptr(env, value, result);
}

napi_status napi_get_value_string_latin1(napi_env env, napi_value value, char* buf, size_t bufsize, size_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_string_latin1_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_string_latin1_ptr(env, value, buf, bufsize, result);
}

napi_status napi_get_value_string_utf16(napi_env env, napi_value value, char16_t* buf, size_t bufsize, size_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_string_utf16_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_string_utf16_ptr(env, value, buf, bufsize, result);
}

napi_status napi_get_value_string_utf8(napi_env env, napi_value value, char* buf, size_t bufsize, size_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_string_utf8_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_string_utf8_ptr(env, value, buf, bufsize, result);
}

napi_status napi_get_value_uint32(napi_env env, napi_value value, uint32_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_uint32_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_uint32_ptr(env, value, result);
}

napi_status napi_get_version(napi_env env, uint32_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_version_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_version_ptr(env, result);
}

napi_status napi_has_element(napi_env env, napi_value object, uint32_t index, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_has_element_ptr) {
        return napi_generic_failure;
    }
    
    return napi_has_element_ptr(env, object, index, result);
}

napi_status napi_has_named_property(napi_env env, napi_value object, const char* utf8name, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_has_named_property_ptr) {
        return napi_generic_failure;
    }
    
    return napi_has_named_property_ptr(env, object, utf8name, result);
}

napi_status napi_has_own_property(napi_env env, napi_value object, napi_value key, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_has_own_property_ptr) {
        return napi_generic_failure;
    }
    
    return napi_has_own_property_ptr(env, object, key, result);
}

napi_status napi_has_property(napi_env env, napi_value object, napi_value key, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_has_property_ptr) {
        return napi_generic_failure;
    }
    
    return napi_has_property_ptr(env, object, key, result);
}

napi_status napi_instanceof(napi_env env, napi_value object, napi_value constructor, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_instanceof_ptr) {
        return napi_generic_failure;
    }
    
    return napi_instanceof_ptr(env, object, constructor, result);
}

napi_status napi_is_array(napi_env env, napi_value value, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_array_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_array_ptr(env, value, result);
}

napi_status napi_is_arraybuffer(napi_env env, napi_value value, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_arraybuffer_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_arraybuffer_ptr(env, value, result);
}

napi_status napi_is_dataview(napi_env env, napi_value value, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_dataview_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_dataview_ptr(env, value, result);
}

napi_status napi_is_error(napi_env env, napi_value value, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_error_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_error_ptr(env, value, result);
}

napi_status napi_is_exception_pending(napi_env env, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_exception_pending_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_exception_pending_ptr(env, result);
}

napi_status napi_is_promise(napi_env env, napi_value value, bool* is_promise) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_promise_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_promise_ptr(env, value, is_promise);
}

napi_status napi_is_typedarray(napi_env env, napi_value value, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_typedarray_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_typedarray_ptr(env, value, result);
}

napi_status napi_new_instance(napi_env env, napi_value constructor, size_t argc, const napi_value* argv, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_new_instance_ptr) {
        return napi_generic_failure;
    }
    
    return napi_new_instance_ptr(env, constructor, argc, argv, result);
}

napi_status napi_open_escapable_handle_scope(napi_env env, napi_escapable_handle_scope* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_open_escapable_handle_scope_ptr) {
        return napi_generic_failure;
    }
    
    return napi_open_escapable_handle_scope_ptr(env, result);
}

napi_status napi_open_handle_scope(napi_env env, napi_handle_scope* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_open_handle_scope_ptr) {
        return napi_generic_failure;
    }
    
    return napi_open_handle_scope_ptr(env, result);
}

napi_status napi_reference_ref(napi_env env, napi_ref ref, uint32_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_reference_ref_ptr) {
        return napi_generic_failure;
    }
    
    return napi_reference_ref_ptr(env, ref, result);
}

napi_status napi_reference_unref(napi_env env, napi_ref ref, uint32_t* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_reference_unref_ptr) {
        return napi_generic_failure;
    }
    
    return napi_reference_unref_ptr(env, ref, result);
}

napi_status napi_reject_deferred(napi_env env, napi_deferred deferred, napi_value rejection) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_reject_deferred_ptr) {
        return napi_generic_failure;
    }
    
    return napi_reject_deferred_ptr(env, deferred, rejection);
}

napi_status napi_remove_wrap(napi_env env, napi_value js_object, void** result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_remove_wrap_ptr) {
        return napi_generic_failure;
    }
    
    return napi_remove_wrap_ptr(env, js_object, result);
}

napi_status napi_resolve_deferred(napi_env env, napi_deferred deferred, napi_value resolution) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_resolve_deferred_ptr) {
        return napi_generic_failure;
    }
    
    return napi_resolve_deferred_ptr(env, deferred, resolution);
}

napi_status napi_run_script(napi_env env, napi_value script, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_run_script_ptr) {
        return napi_generic_failure;
    }
    
    return napi_run_script_ptr(env, script, result);
}

napi_status napi_set_element(napi_env env, napi_value object, uint32_t index, napi_value value) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_set_element_ptr) {
        return napi_generic_failure;
    }
    
    return napi_set_element_ptr(env, object, index, value);
}

napi_status napi_set_named_property(napi_env env, napi_value object, const char* utf8name, napi_value value) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_set_named_property_ptr) {
        return napi_generic_failure;
    }
    
    return napi_set_named_property_ptr(env, object, utf8name, value);
}

napi_status napi_set_property(napi_env env, napi_value object, napi_value key, napi_value value) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_set_property_ptr) {
        return napi_generic_failure;
    }
    
    return napi_set_property_ptr(env, object, key, value);
}

napi_status napi_strict_equals(napi_env env, napi_value lhs, napi_value rhs, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_strict_equals_ptr) {
        return napi_generic_failure;
    }
    
    return napi_strict_equals_ptr(env, lhs, rhs, result);
}

napi_status napi_throw(napi_env env, napi_value error) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_throw_ptr) {
        return napi_generic_failure;
    }
    
    return napi_throw_ptr(env, error);
}

napi_status napi_throw_error(napi_env env, const char* code, const char* msg) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_throw_error_ptr) {
        return napi_generic_failure;
    }
    
    return napi_throw_error_ptr(env, code, msg);
}

napi_status napi_throw_range_error(napi_env env, const char* code, const char* msg) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_throw_range_error_ptr) {
        return napi_generic_failure;
    }
    
    return napi_throw_range_error_ptr(env, code, msg);
}

napi_status napi_throw_type_error(napi_env env, const char* code, const char* msg) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_throw_type_error_ptr) {
        return napi_generic_failure;
    }
    
    return napi_throw_type_error_ptr(env, code, msg);
}

napi_status napi_typeof(napi_env env, napi_value value, napi_valuetype* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_typeof_ptr) {
        return napi_generic_failure;
    }
    
    return napi_typeof_ptr(env, value, result);
}

napi_status napi_unwrap(napi_env env, napi_value js_object, void** result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_unwrap_ptr) {
        return napi_generic_failure;
    }
    
    return napi_unwrap_ptr(env, js_object, result);
}

napi_status napi_wrap(napi_env env, napi_value js_object, void* native_object, napi_finalize finalize_cb, void* finalize_hint, napi_ref* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_wrap_ptr) {
        return napi_generic_failure;
    }
    
    return napi_wrap_ptr(env, js_object, native_object, finalize_cb, finalize_hint, result);
}

napi_status napi_async_destroy(napi_env env, napi_async_context async_context) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_async_destroy_ptr) {
        return napi_generic_failure;
    }
    
    return napi_async_destroy_ptr(env, async_context);
}

napi_status napi_async_init(napi_env env, napi_value async_resource, napi_value async_resource_name, napi_async_context* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_async_init_ptr) {
        return napi_generic_failure;
    }
    
    return napi_async_init_ptr(env, async_resource, async_resource_name, result);
}

napi_status napi_cancel_async_work(napi_env env, napi_async_work work) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_cancel_async_work_ptr) {
        return napi_generic_failure;
    }
    
    return napi_cancel_async_work_ptr(env, work);
}

napi_status napi_create_async_work(napi_env env, napi_value async_resource, napi_value async_resource_name, napi_async_execute_callback execute, napi_async_complete_callback complete, void* data, napi_async_work* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_async_work_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_async_work_ptr(env, async_resource, async_resource_name, execute, complete, data, result);
}

napi_status napi_create_buffer(napi_env env, size_t length, void** data, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_buffer_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_buffer_ptr(env, length, data, result);
}

napi_status napi_create_buffer_copy(napi_env env, size_t length, const void* data, void** result_data, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_buffer_copy_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_buffer_copy_ptr(env, length, data, result_data, result);
}

napi_status napi_create_external_buffer(napi_env env, size_t length, void* data, napi_finalize finalize_cb, void* finalize_hint, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_external_buffer_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_external_buffer_ptr(env, length, data, finalize_cb, finalize_hint, result);
}

napi_status napi_delete_async_work(napi_env env, napi_async_work work) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_delete_async_work_ptr) {
        return napi_generic_failure;
    }
    
    return napi_delete_async_work_ptr(env, work);
}

void napi_fatal_error(const char* location, size_t location_len, const char* message, size_t message_len) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return;
    }
    
    if (!napi_fatal_error_ptr) {
        return;
    }
    
    napi_fatal_error_ptr(location, location_len, message, message_len);
}

napi_status napi_get_buffer_info(napi_env env, napi_value value, void** data, size_t* length) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_buffer_info_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_buffer_info_ptr(env, value, data, length);
}

napi_status napi_get_node_version(napi_env env, const napi_node_version** version) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_node_version_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_node_version_ptr(env, version);
}

napi_status napi_is_buffer(napi_env env, napi_value value, bool* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_buffer_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_buffer_ptr(env, value, result);
}

napi_status napi_make_callback(napi_env env, napi_async_context async_context, napi_value recv, napi_value func, size_t argc, const napi_value* argv, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_make_callback_ptr) {
        return napi_generic_failure;
    }
    
    return napi_make_callback_ptr(env, async_context, recv, func, argc, argv, result);
}

void napi_module_register(napi_module* mod) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return;
    }
    
    if (!napi_module_register_ptr) {
        return;
    }
    
    napi_module_register_ptr(mod);
}

napi_status napi_queue_async_work(napi_env env, napi_async_work work) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_queue_async_work_ptr) {
        return napi_generic_failure;
    }
    
    return napi_queue_async_work_ptr(env, work);
}

napi_status napi_get_uv_event_loop(napi_env env, struct uv_loop_s** loop) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_uv_event_loop_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_uv_event_loop_ptr(env, loop);
}

napi_status napi_add_env_cleanup_hook(napi_env env, napi_cleanup_hook fun, void* arg) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_add_env_cleanup_hook_ptr) {
        return napi_generic_failure;
    }
    
    return napi_add_env_cleanup_hook_ptr(env, fun, arg);
}

napi_status napi_close_callback_scope(napi_env env, napi_callback_scope scope) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_close_callback_scope_ptr) {
        return napi_generic_failure;
    }
    
    return napi_close_callback_scope_ptr(env, scope);
}

napi_status napi_fatal_exception(napi_env env, napi_value err) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_fatal_exception_ptr) {
        return napi_generic_failure;
    }
    
    return napi_fatal_exception_ptr(env, err);
}

napi_status napi_open_callback_scope(napi_env env, napi_value resource_object, napi_async_context context, napi_callback_scope* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_open_callback_scope_ptr) {
        return napi_generic_failure;
    }
    
    return napi_open_callback_scope_ptr(env, resource_object, context, result);
}

napi_status napi_remove_env_cleanup_hook(napi_env env, napi_cleanup_hook fun, void* arg) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_remove_env_cleanup_hook_ptr) {
        return napi_generic_failure;
    }
    
    return napi_remove_env_cleanup_hook_ptr(env, fun, arg);
}

napi_status napi_acquire_threadsafe_function(napi_threadsafe_function func) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_acquire_threadsafe_function_ptr) {
        return napi_generic_failure;
    }
    
    return napi_acquire_threadsafe_function_ptr(func);
}

napi_status napi_call_threadsafe_function(napi_threadsafe_function func, void* data, napi_threadsafe_function_call_mode is_blocking) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_call_threadsafe_function_ptr) {
        return napi_generic_failure;
    }
    
    return napi_call_threadsafe_function_ptr(func, data, is_blocking);
}

napi_status napi_create_threadsafe_function(napi_env env, napi_value func, napi_value async_resource, napi_value async_resource_name, size_t max_queue_size, size_t initial_thread_count, void* thread_finalize_data, napi_finalize thread_finalize_cb, void* context, napi_threadsafe_function_call_js call_js_cb, napi_threadsafe_function* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_threadsafe_function_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_threadsafe_function_ptr(env, func, async_resource, async_resource_name, max_queue_size, initial_thread_count, thread_finalize_data, thread_finalize_cb, context, call_js_cb, result);
}

napi_status napi_get_threadsafe_function_context(napi_threadsafe_function func, void** result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_threadsafe_function_context_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_threadsafe_function_context_ptr(func, result);
}

napi_status napi_ref_threadsafe_function(napi_env env, napi_threadsafe_function func) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_ref_threadsafe_function_ptr) {
        return napi_generic_failure;
    }
    
    return napi_ref_threadsafe_function_ptr(env, func);
}

napi_status napi_release_threadsafe_function(napi_threadsafe_function func, napi_threadsafe_function_release_mode mode) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_release_threadsafe_function_ptr) {
        return napi_generic_failure;
    }
    
    return napi_release_threadsafe_function_ptr(func, mode);
}

napi_status napi_unref_threadsafe_function(napi_env env, napi_threadsafe_function func) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_unref_threadsafe_function_ptr) {
        return napi_generic_failure;
    }
    
    return napi_unref_threadsafe_function_ptr(env, func);
}

napi_status napi_add_finalizer(napi_env env, napi_value js_object, void* finalize_data, napi_finalize finalize_cb, void* finalize_hint, napi_ref* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_add_finalizer_ptr) {
        return napi_generic_failure;
    }
    
    return napi_add_finalizer_ptr(env, js_object, finalize_data, finalize_cb, finalize_hint, result);
}

napi_status napi_create_date(napi_env env, double time, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_date_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_date_ptr(env, time, result);
}

napi_status napi_get_date_value(napi_env env, napi_value value, double* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_date_value_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_date_value_ptr(env, value, result);
}

napi_status napi_is_date(napi_env env, napi_value value, bool* is_date) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_is_date_ptr) {
        return napi_generic_failure;
    }
    
    return napi_is_date_ptr(env, value, is_date);
}

napi_status napi_create_bigint_int64(napi_env env, int64_t value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_bigint_int64_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_bigint_int64_ptr(env, value, result);
}

napi_status napi_create_bigint_uint64(napi_env env, uint64_t value, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_bigint_uint64_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_bigint_uint64_ptr(env, value, result);
}

napi_status napi_create_bigint_words(napi_env env, int sign_bit, size_t word_count, const uint64_t* words, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_create_bigint_words_ptr) {
        return napi_generic_failure;
    }
    
    return napi_create_bigint_words_ptr(env, sign_bit, word_count, words, result);
}

napi_status napi_get_all_property_names(napi_env env, napi_value object, napi_key_collection_mode key_mode, napi_key_filter key_filter, napi_key_conversion key_conversion, napi_value* result) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_all_property_names_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_all_property_names_ptr(env, object, key_mode, key_filter, key_conversion, result);
}

napi_status napi_get_instance_data(napi_env env, void** data) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_instance_data_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_instance_data_ptr(env, data);
}

napi_status napi_get_value_bigint_int64(napi_env env, napi_value value, int64_t* result, bool* lossless) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_bigint_int64_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_bigint_int64_ptr(env, value, result, lossless);
}

napi_status napi_get_value_bigint_uint64(napi_env env, napi_value value, uint64_t* result, bool* lossless) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_bigint_uint64_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_bigint_uint64_ptr(env, value, result, lossless);
}

napi_status napi_get_value_bigint_words(napi_env env, napi_value value, int* sign_bit, size_t* word_count, uint64_t* words) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_get_value_bigint_words_ptr) {
        return napi_generic_failure;
    }
    
    return napi_get_value_bigint_words_ptr(env, value, sign_bit, word_count, words);
}

napi_status napi_set_instance_data(napi_env env, void* data, napi_finalize finalize_cb, void* finalize_hint) {
    if (!napi_functions_loaded && !discover_and_load_napi_functions()) {
        return napi_generic_failure;
    }
    
    if (!napi_set_instance_data_ptr) {
        return napi_generic_failure;
    }
    
    return napi_set_instance_data_ptr(env, data, finalize_cb, finalize_hint);
}

__attribute__((constructor))
static void init_napi_dynamic_load() {
    discover_and_load_napi_functions();
}

__attribute__((destructor))
static void cleanup_napi_dynamic_load() {
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        *(napi_functions[i].func_ptr) = NULL;
    }
    
    napi_module_handle = NULL;
    napi_functions_loaded = 0;
}

int napi_dynamic_load_init() {
    return discover_and_load_napi_functions();
}

int napi_dynamic_load_is_loaded() {
    return napi_functions_loaded;
}

int napi_dynamic_load_get_function_count() {
    int count = 0;
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        if (*(napi_functions[i].func_ptr) != NULL) {
            count++;
        }
    }
    return count;
}

int napi_dynamic_load_is_function_available(const char* function_name) {
    if (!function_name) return 0;
    
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        if (strcmp(napi_functions[i].name, function_name) == 0) {
            return *(napi_functions[i].func_ptr) != NULL;
        }
    }
    
    return 0;
}

const char* napi_dynamic_load_get_version(void) {
    return "1.0.0 (NAPI v6 compatible)";
}

int napi_dynamic_load_get_status_info(char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return -1;
    }
    
    char temp[2048] = {0};
    int len = 0;
    
    len += snprintf(temp + len, sizeof(temp) - len, 
        "NAPI Dynamic Load Status:\n");
    len += snprintf(temp + len, sizeof(temp) - len, 
        "  Loaded: %s\n", napi_functions_loaded ? "Yes" : "No");
    len += snprintf(temp + len, sizeof(temp) - len, 
        "  Module Handle: %p\n", (void*)napi_module_handle);
    
    int total_functions = 0;
    int loaded_functions = 0;
    
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        total_functions++;
        if (*(napi_functions[i].func_ptr) != NULL) {
            loaded_functions++;
        }
    }
    
    len += snprintf(temp + len, sizeof(temp) - len, 
        "  Functions: %d/%d loaded\n", loaded_functions, total_functions);
    
    if (len >= buffer_size) {
        return -((int)buffer_size);
    }
    
    strcpy(buffer, temp);
    return len;
}

int napi_dynamic_load_reload(void) {
    napi_functions_loaded = 0;
    napi_module_handle = NULL;
    
    for (int i = 0; napi_functions[i].name != NULL; i++) {
        *(napi_functions[i].func_ptr) = NULL;
    }
    
    return discover_and_load_napi_functions();
}

__attribute__((visibility("default")))
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
  switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
      napi_dynamic_load_init();
      break;
  }
  return TRUE;
}

#endif // _WIN32
