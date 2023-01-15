<!-- index.php, Jake Deery, 2018 -->
<!DOCTYPE html>
<HTML LANG='en'>
    <HEAD>
        <TITLE>SDFD203 BY JAKE DEERY | HOMEPAGE</TITLE>
        <META CHARSET='utf-8' />
        <STYLE TYPE='text/css'>
        th {
            text-align: left
        }

        td, th {
            min-width: 100px;
            min-height: 100px
        }
        </STYLE>
    </HEAD>
	<BODY>
        <H1>SDFD203 BY JAKE DEERY | HOMEPAGE</H1>
        <P STYLE='font-size:8.5pt'>Last updated: 18th April 2018</P>
        <P>Begin by searching a string, or click <A HREF='admin.php'>here</A> to administrate.</p>
        <HR />
        <FORM ACTION='index.php' METHOD='get'>
		    <INPUT TYPE='text' NAME='string' />
		    <BUTTON TYPE='submit'>Search</BUTTON>
        </FORM>
        <PRE><?php /* php include files */ if(isset($_GET["string"])) { include "modules/search_go.php"; search_go(); } ?></PRE>
    </BODY>
</HTML>
