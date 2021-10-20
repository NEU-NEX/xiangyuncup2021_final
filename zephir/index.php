<?php
session_start();
echo "<h1>Welcome to zep online testing system.</h1>";
highlight_file(__FILE__);
function generateRandomString($length = 10) {
    $characters = 'abcdefghijklmnopqrstuvwxyz';
    $charactersLength = strlen($characters);
    $randomString = '';
    for ($i = 0; $i < $length; $i++) {
        $randomString .= $characters[rand(0, $charactersLength - 1)];
    }
    return $randomString;
}
function create_sandbox(){
    $sandbox_dir = "/tmp/".md5("eqe".$_SERVER["REMOTE_ADDR"].$_SERVER['HTTP_USER_AGENT']."aqa")."/";
    if(is_dir($sandbox_dir)){
        return;
    }
    $res = mkdir($sandbox_dir, 0777, true);
    if($res) {
        $_SESSION['sandbox_dir'] = $sandbox_dir;
        echo $sandbox_dir;
    }
    else{
        echo "something error";
    }
} 
function init_proj(){
    if(!isset($_SESSION['sandbox_dir'])){
        die("error");
    }
    $namespace = "C".generateRandomString(7);
    $class = generateRandomString(8);
    $_SESSION['namespace'] = $namespace;
    $_SESSION['class'] = $class;
    file("http://127.0.0.1:8080/api.php?c=init&namespace=".$namespace."&class=".$class);
}
$tmpl = <<<ZEPF
namespace {namespace};

class {class}{
    public function test(){
        return "{string}";
    }
}
ZEPF;
$php_ini_tmpl = <<<PHPINI
disable_functions = zend_version, func_num_args, func_get_arg, func_get_args, strlen, strcmp, strncmp, strcasecmp, strncasecmp, error_reporting, define, defined, get_class, get_called_class, get_parent_class, is_subclass_of, is_a, get_class_vars, get_object_vars, get_mangled_object_vars, get_class_methods, method_exists, property_exists, class_exists, interface_exists, trait_exists, enum_exists, function_exists, class_alias, get_included_files, get_required_files, trigger_error, user_error, set_error_handler, restore_error_handler, set_exception_handler, restore_exception_handler, get_declared_classes, get_declared_traits, get_declared_interfaces, get_defined_functions, get_defined_vars, get_resource_type, get_resource_id, get_resources, get_loaded_extensions, get_defined_constants, debug_backtrace, debug_print_backtrace, extension_loaded, get_extension_funcs, gc_mem_caches, gc_collect_cycles, gc_enabled, gc_enable, gc_disable, gc_status, strtotime, date, idate, gmdate, mktime, gmmktime, checkdate, strftime, gmstrftime, time, localtime, getdate, date_create, date_create_immutable, date_create_from_format, date_create_immutable_from_format, date_parse, date_parse_from_format, date_get_last_errors, date_format, date_modify, date_add, date_sub, date_timezone_get, date_timezone_set, date_offset_get, date_diff, date_time_set, date_date_set, date_isodate_set, date_timestamp_set, date_timestamp_get, timezone_open, timezone_name_get, timezone_name_from_abbr, timezone_offset_get, timezone_transitions_get, timezone_location_get, timezone_identifiers_list, timezone_abbreviations_list, timezone_version_get, date_interval_create_from_date_string, date_interval_format, date_default_timezone_set, date_default_timezone_get, date_sunrise, date_sunset, date_sun_info, preg_match, preg_match_all, preg_replace, preg_filter, preg_replace_callback, preg_replace_callback_array, preg_split, preg_quote, preg_grep, preg_last_error, preg_last_error_msg, hash, hash_file, hash_hmac, hash_hmac_file, hash_init, hash_update, hash_update_stream, hash_update_file, hash_final, hash_copy, hash_algos, hash_hmac_algos, hash_pbkdf2, hash_equals, hash_hkdf, json_encode, json_decode, json_last_error, json_last_error_msg, class_implements, class_parents, class_uses, spl_autoload, spl_autoload_call, spl_autoload_extensions, spl_autoload_functions, spl_autoload_register, spl_autoload_unregister, spl_classes, spl_object_hash, spl_object_id, iterator_apply, iterator_count, iterator_to_array, set_time_limit, header_register_callback, ob_start, ob_flush, ob_clean, ob_end_flush, ob_end_clean, ob_get_flush, ob_get_clean, ob_get_contents, ob_get_level, ob_get_length, ob_list_handlers, ob_get_status, ob_implicit_flush, output_reset_rewrite_vars, output_add_rewrite_var, stream_wrapper_register, stream_register_wrapper, stream_wrapper_unregister, stream_wrapper_restore, array_push, krsort, ksort, count, sizeof, natsort, natcasesort, asort, arsort, sort, rsort, usort, uasort, uksort, end, prev, next, reset, current, pos, key, min, max, array_walk, array_walk_recursive, in_array, array_search, extract, compact, array_fill, array_fill_keys, range, shuffle, array_pop, array_shift, array_unshift, array_splice, array_slice, array_merge, array_merge_recursive, array_replace, array_replace_recursive, array_keys, array_key_first, array_key_last, array_values, array_count_values, array_column, array_reverse, array_pad, array_flip, array_change_key_case, array_unique, array_intersect_key, array_intersect_ukey, array_intersect, array_uintersect, array_intersect_assoc, array_uintersect_assoc, array_intersect_uassoc, array_uintersect_uassoc, array_diff_key, array_diff_ukey, array_diff, array_udiff, array_diff_assoc, array_diff_uassoc, array_udiff_assoc, array_udiff_uassoc, array_multisort, array_rand, array_sum, array_product, array_reduce, array_filter, array_map, array_key_exists, key_exists, array_chunk, array_combine, array_is_list, base64_encode, base64_decode, constant, ip2long, long2ip, getenv, putenv, getopt, flush, sleep, usleep, time_nanosleep, time_sleep_until, get_current_user, get_cfg_var, error_log, error_get_last, error_clear_last, call_user_func, call_user_func_array, forward_static_call, forward_static_call_array, register_shutdown_function, highlight_file, show_source, php_strip_whitespace, highlight_string, ini_get, ini_get_all, ini_set, ini_alter, ini_restore, set_include_path, get_include_path, print_r, connection_aborted, connection_status, ignore_user_abort, getservbyname, getservbyport, getprotobyname, getprotobynumber, register_tick_function, unregister_tick_function, is_uploaded_file, move_uploaded_file, parse_ini_file, parse_ini_string, sys_getloadavg, get_browser, crc32, crypt, strptime, gethostname, gethostbyaddr, gethostbyname, gethostbynamel, dns_check_record, checkdnsrr, dns_get_record, dns_get_mx, getmxrr, net_get_interfaces, ftok, hrtime, lcg_value, md5, md5_file, getmyuid, getmygid, getmypid, getmyinode, getlastmod, sha1, sha1_file, openlog, closelog, syslog, inet_ntop, inet_pton, metaphone, header, header_remove, setrawcookie, setcookie, http_response_code, headers_sent, headers_list, htmlspecialchars, htmlspecialchars_decode, html_entity_decode, htmlentities, get_html_translation_table, assert, assert_options, bin2hex, hex2bin, strspn, strcspn, nl_langinfo, strcoll, trim, rtrim, chop, ltrim, wordwrap, explode, implode, join, strtok, strtoupper, strtolower, basename, dirname, pathinfo, stristr, strstr, strchr, strpos, stripos, strrpos, strripos, strrchr, str_contains, str_starts_with, str_ends_with, chunk_split, substr, substr_replace, quotemeta, ucfirst, lcfirst, ucwords, strtr, strrev, similar_text, addcslashes, addslashes, stripcslashes, stripslashes, str_replace, str_ireplace, hebrev, nl2br, strip_tags, setlocale, parse_str, str_getcsv, count_chars, strnatcmp, localeconv, strnatcasecmp, substr_count, str_pad, sscanf, str_rot13, str_shuffle, str_word_count, str_split, strpbrk, substr_compare, utf8_encode, utf8_decode, opendir, dir, closedir, chdir, chroot, getcwd, rewinddir, readdir, scandir, glob, exec, system, passthru, escapeshellcmd, escapeshellarg, shell_exec, proc_nice, flock, get_meta_tags, pclose, popen, readfile, rewind, rmdir, umask, fclose, feof, fgetc, fgets, fread, fopen, fscanf, fpassthru, ftruncate, fstat, fseek, ftell, fflush, fsync, fdatasync, fwrite, fputs, mkdir, rename, copy, tempnam, tmpfile, file, file_get_contents, unlink, file_put_contents, fputcsv, fgetcsv, realpath, fnmatch, sys_get_temp_dir, fileatime, filectime, filegroup, fileinode, filemtime, fileowner, fileperms, filesize, filetype, file_exists, is_writable, is_writeable, is_readable, is_executable, is_file, is_dir, is_link, stat, lstat, chown, chgrp, lchown, lchgrp, chmod, touch, clearstatcache, disk_total_space, disk_free_space, diskfreespace, realpath_cache_get, realpath_cache_size, sprintf, printf, vprintf, vsprintf, fprintf, vfprintf, fsockopen, pfsockopen, http_build_query, image_type_to_mime_type, image_type_to_extension, getimagesize, getimagesizefromstring, phpinfo, phpversion, phpcredits, php_sapi_name, php_uname, php_ini_scanned_files, php_ini_loaded_file, iptcembed, iptcparse, levenshtein, readlink, linkinfo, symlink, link, mail, abs, ceil, floor, round, sin, cos, tan, asin, acos, atan, atanh, atan2, sinh, cosh, tanh, asinh, acosh, expm1, log1p, pi, is_finite, is_nan, intdiv, is_infinite, pow, exp, log, log10, sqrt, hypot, deg2rad, rad2deg, bindec, hexdec, octdec, decbin, base_convert, number_format, fmod, fdiv, microtime, gettimeofday, getrusage, pack, unpack, password_get_info, password_hash, password_needs_rehash, password_verify, password_algos, proc_open, proc_close, proc_terminate, proc_get_status, quoted_printable_decode, quoted_printable_encode, mt_srand, srand, rand, mt_rand, mt_getrandmax, getrandmax, random_bytes, random_int, soundex, stream_select, stream_context_create, stream_context_set_params, stream_context_get_params, stream_context_set_option, stream_context_get_options, stream_context_get_default, stream_context_set_default, stream_filter_prepend, stream_filter_append, stream_filter_remove, stream_socket_client, stream_socket_server, stream_socket_accept, stream_socket_get_name, stream_socket_recvfrom, stream_socket_sendto, stream_socket_enable_crypto, stream_socket_shutdown, stream_socket_pair, stream_copy_to_stream, stream_get_contents, stream_supports_lock, stream_set_write_buffer, set_file_buffer, stream_set_read_buffer, stream_set_blocking, socket_set_blocking, stream_get_meta_data, socket_get_status, stream_get_line, stream_resolve_include_path, stream_get_wrappers, stream_get_transports, stream_is_local, stream_isatty, stream_set_chunk_size, stream_set_timeout, socket_set_timeout, gettype, get_debug_type, settype, intval, floatval, doubleval, boolval, strval, is_null, is_resource, is_bool, is_int, is_integer, is_long, is_float, is_double, is_numeric, is_string, is_array, is_object, is_scalar, is_callable, is_iterable, is_countable, uniqid, parse_url, urlencode, urldecode, rawurlencode, rawurldecode, get_headers, stream_bucket_make_writeable, stream_bucket_prepend, stream_bucket_append, stream_bucket_new, stream_get_filters, stream_filter_register, convert_uuencode, convert_uudecode, var_dump, var_export, debug_zval_dump, serialize, unserialize, memory_get_usage, memory_get_peak_usage, version_compare, dl, cli_set_process_title, cli_get_process_title
disable_classes = SplFileObject,Exception,SplDoublyLinkedList,Error,ErrorException,ArgumentCountError,ArithmeticError,AssertionError,DivisionByZeroError,CompileError,ParseError,TypeError,ValueError,UnhandledMatchError,ClosedGeneratorException,LogicException,BadFunctionCallException,BadMethodCallException,DomainException,InvalidArgumentException,LengthException,OutOfRangeException,PharException,ReflectionException,RuntimeException,OutOfBoundsException,OverflowException,PDOException,RangeException,UnderflowException,UnexpectedValueException,JsonException,SodiumException   Exception,SplDoublyLinkedLit,Error,ErrorException,ArgumentCountError,ArithmeticError,AsserttionError,DivisionByZeroError,CompileError,ParseError,TypeError,ValueError,UnhandledMatchError,ClosedGeneratorException,LogicException,BadFunctionCallException,BadMethodCallException,DomainException,InvalidArgumentException,LengthException,OutOfRangeException,PharException,ReflectionException,RuntimeException,OutOfBoundsException,OverflowException,PDOException,RangeException,UnderflowException,UnexpectedValueException,JsonException,SodiumException
open_basedir = /tmp/
extension = {extension}
PHPINI;

$exec_file_tmpl = <<<INDEXPHP
<?php
\$c = new \\{namespace}\\{class};
echo \$c->test();
?>
INDEXPHP;

function uploadzep(){
    global $php_ini_tmpl,$exec_file_tmpl,$tmpl;
    if(!isset($_POST['zep'])){
        die("upload");
    }
    if(!isset($_SESSION['namespace'])||!isset($_SESSION['class'])){
        die("error");
    }
    if(!isset($_SESSION['sandbox_dir'])){
        die("error");
    }
    $zepstring = addslashes($_POST['zep']);
    $namespace = $_SESSION['namespace'];
    $class = $_SESSION['class'];
    $zep_file = preg_replace('/(.*)\{namespace\}(.*)/is', '${1}'.$namespace.'${2}', $tmpl);
    $zep_file = preg_replace('/(.*)\{class\}(.*)/is', '${1}'.$class.'${2}', $zep_file);
    $zep_file = preg_replace('/(.*)\{string\}(.*)/is', '${1}'.$zepstring.'${2}', $zep_file);
    file_put_contents("/tmp/".md5($zep_file), $zep_file);
    file("http://127.0.0.1:8080/api.php?c=create&namespace=".$namespace."&hash=".md5($zep_file)."&class=".$class);
    $extension_path = "/tmp/".strtolower($namespace)."/ext/modules/".strtolower($namespace).".so";
    $php_ini = preg_replace('/(.*)\{extension\}(.*)/is', '${1}'.$extension_path.'${2}', $php_ini_tmpl);
    $php_ini_path = $_SESSION['sandbox_dir'].md5($zep_file.$php_ini);
    $exec_file_content = preg_replace('/(.*)\{namespace\}(.*)/is', '${1}'.$namespace.'${2}', $exec_file_tmpl);
    $exec_file_content = preg_replace('/(.*)\{class\}(.*)/is', '${1}'.$class.'${2}', $exec_file_content);
    $exec_path = $_SESSION['sandbox_dir'].md5($zep_file.$exec_file_content);
    file_put_contents($php_ini_path, $php_ini);
    file_put_contents($exec_path, $exec_file_content);
    echo "<br>------------result------------<br>";
    system("php -c ".$php_ini_path." ".$exec_path);
    echo "<br>------------------------------<br>";
    system("rm -rf ".$_SESSION['sandbox_dir']);
    @unlink("/tmp/".md5($zep_file));
    file("http://127.0.0.1:8080/api.php?c=delete&namespace=".$namespace);
}
create_sandbox();
if(isset($_GET['c'])){
    switch($_GET['c']){
        case 'init':
            init_proj();
            break;
        case 'upload':
            uploadzep();
            break;
        default:
            break;
    }
}

?> 