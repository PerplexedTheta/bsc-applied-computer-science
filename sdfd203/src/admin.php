<!-- admin.php, Jake Deery, 2018 -->
<!DOCTYPE html>
<HTML LANG='en'>
	<HEAD>
		<TITLE>SDFD203 BY JAKE DEERY | ADMINISTRATE</TITLE>
		<META CHARSET='utf-8' />
	</HEAD>
	<BODY>
		<H1>SDFD203 BY JAKE DEERY | ADMINISTRATE</H1>
		<P STYLE='font-size:8.5pt'>Last updated: 17th April 2018</P>
		<P>Begin by chosing an option, or click <A HREF='./'>here</A> to go back.</p>
		<HR />
		<UL>
			<LI>
				<FORM ACTION='admin.php' METHOD='get'>
					<BUTTON TYPE='submit' name='check_db'>Check DB</BUTTON>
					 - <SPAN STYLE='font-style:italic'>Check to see the database works or not</SPAN>
				</FORM>
			</LI>
			<LI>
				<FORM ACTION='admin.php' METHOD='get'>
					<BUTTON TYPE='submit' NAME='truncate'>Truncate</BUTTON>
					 - <SPAN STYLE='font-style:italic'>Completely erase the contents of the search schema in the database.</SPAN>
				</FORM>
			</LI>
			<LI>
				<FORM ACTION='admin.php' METHOD='get'>
					<BUTTON TYPE='submit' NAME='update'>Update</BUTTON>
					 - <SPAN STYLE='font-style:italic'>Update the search schema in the database</SPAN>
				</FORM>
			</LI>
		</UL>
		<PRE><?php /* php include files */ include "modules/calls.php"; ?></PRE>
	</BODY>
</HTML>
