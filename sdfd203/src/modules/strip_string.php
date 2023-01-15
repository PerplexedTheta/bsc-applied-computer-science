<?php
// modules/strip_string.php, Jake Deery, 2018

// strip punctuation and lower string(s)
function strip_string($input_str) {
	if ($input_str == "*") {
		// skip this step
	} else {
		$input_str = strtolower($input_str);
		$input_str = preg_replace("/[^a-zA-Z]/", "", $input_str);
		$input_str = trim($input_str);
	}

	return $input_str;
}
?>