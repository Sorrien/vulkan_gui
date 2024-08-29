use ash::ext::debug_utils::Instance as DebugUtils;
use ash::{vk, Entry};
use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::Arc;

pub struct DebugMessenger {
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub debug_utils: DebugUtils,
    use_debug: bool,
}

impl DebugMessenger {
    pub fn new(
        entry: &Entry,
        instance: Arc<crate::ash_bootstrap::Instance>,
        use_debug: bool,
    ) -> Self {
        let (debug_utils, debug_messenger) = debug_utils(entry, &instance.handle, use_debug);
        Self {
            debug_messenger,
            debug_utils,
            use_debug,
        }
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        if self.use_debug {
            unsafe {
                self.debug_utils
                    .destroy_debug_utils_messenger(self.debug_messenger, None)
            };
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    if message_severity > vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );
    }

    vk::FALSE
}

pub fn debug_utils(
    entry: &Entry,
    instance: &ash::Instance,
    use_debug: bool,
) -> (DebugUtils, vk::DebugUtilsMessengerEXT) {
    let debug_utils_loader = DebugUtils::new(entry, &instance);

    let debug_call_back = if use_debug {
        create_debug_callback(&debug_utils_loader)
    } else {
        ash::vk::DebugUtilsMessengerEXT::null()
    };

    (debug_utils_loader, debug_call_back)
}

fn create_debug_callback(debug_utils_loader: &DebugUtils) -> ash::vk::DebugUtilsMessengerEXT {
    let debug_info = create_debug_info();

    unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_info, None)
            .unwrap()
    }
}

pub fn create_debug_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    let x = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    x
}

/* pub fn set_debug_utils_object_name<T: vk::Handle>(
    debug_utils_loader: &DebugUtils,
    device: vk::Device,
    object_handle: T,
    object_name: &str,
) {
    let name_cstr = std::ffi::CString::new(object_name).expect("wrong string parameter");

    let mut name_info = vk::DebugUtilsObjectNameInfoEXT::default()
        .object_handle(object_handle)
        .object_name(&name_cstr);
    name_info.s_type = StructureType::from_raw(T::TYPE.as_raw());

    let _ = unsafe { debug_utils_loader.set_debug_utils_object_name(device, &name_info) };
} */
