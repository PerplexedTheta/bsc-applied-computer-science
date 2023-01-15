<?php
// modules/search_go.php, Jake Deery, 2018
if(!function_exists("egg")) {
	include "egg.php";
}

if(!function_exists("strip_string")) {
	include "strip_string.php";
}

if(!function_exists("read_file")) {
	include "read_file.php";
}

if(!function_exists("retrieve")) {
	include "retrieve.php";
}

function search_go() {
	// load the search string into an array
	$string = $_GET["string"];
	printf("Searched for: " . $string . "\n\n");

	if ($string == "chinchin") {
		egg();
		exit();
	}

	$search_arr = explode(" ", $string);
	// here, we'll want to remove useless punctuation and lower it
	foreach ($search_arr as $key => $search_value) {
		$search_arr[$key] = strip_string($search_value);
	}

	$arr = retrieve($search_arr);
	printf("Restuls:");
	// table
	printf("<TABLE>\n<TBODY>\n");
	printf("<TR>\n<TH>ID</TH>\n<TH>Keywords</TH>\n<TH>Filename</TH>\n</TR>\n");
	for ($i = 0; $i < sizeof($arr["id"]); $i++) {
		printf("<TR>\n<TD>" . $arr["id"][$i] . "</TD>\n<TD>" . $arr["key"][$i] . "</TD>\n<TD><A HREF='search_source/" . $arr["dir"][$i] . "' TARGET='_blank'>" . $arr["dir"][$i] . "</A></TD>\n</TR>\n");
	}
	printf("<TR>\n<TD>&nbsp;</TD>\n<TD>&nbsp;</TD>\n<TD>&nbsp;</TD>\n</TR>\n");
	printf("</TBODY>\n</TABLE>");

	// log
	//printf("log output\n");
	//print_r($arr);
}
?>
